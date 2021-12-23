using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Telegram.Bot;
using Telegram.Bot.Exceptions;
using Telegram.Bot.Extensions.Polling;
using Telegram.Bot.Types;
using Telegram.Bot.Types.Enums;

namespace AIMLTGBot
{
    public class TelegramService : IDisposable
    {
        private readonly TelegramBotClient client;
        private readonly AIMLService aiml;
        private readonly NeuralNetwork net;
        private MagicEye processor = new MagicEye();
        private int symbolsCount = 10;

        // CancellationToken - инструмент для отмены задач, запущенных в отдельном потоке
        private readonly CancellationTokenSource cts = new CancellationTokenSource();
        public string Username { get; }

        public TelegramService(string token, AIMLService aimlService)
        {
            aiml = aimlService;
            client = new TelegramBotClient(token);
            client.StartReceiving(HandleUpdateMessageAsync, HandleErrorAsync, new ReceiverOptions
            {   // Подписываемся только на сообщения
                AllowedUpdates = new[] { UpdateType.Message }
            },
            cancellationToken: cts.Token);
            // Пробуем получить логин бота - тестируем соединение и токен
            Username = client.GetMeAsync().Result.Username;

            net = new NeuralNetwork();

            processor.settings.threshold = 127;
            processor.settings.differenceLim = 0.15f;
        }

        async Task HandleUpdateMessageAsync(ITelegramBotClient botClient, Update update, CancellationToken cancellationToken)
        {
            var message = update.Message;
            var chatId = message.Chat.Id;
            var username = message.Chat.FirstName;
            if (message.Type == MessageType.Text)
            {
                var messageText = update.Message.Text;

                Console.WriteLine($"Received a '{messageText}' message in chat {chatId} with {username}.");

                var answer = aiml.Talk(chatId, username, messageText);
                await botClient.SendTextMessageAsync(
                    chatId: chatId,
                    text: answer.Substring(0, answer.Length-1),
                    cancellationToken: cancellationToken);
                return;
            }
            // Загрузка изображений пригодится для соединения с нейросетью
            if (message.Type == MessageType.Photo)
            {
                var photoId = message.Photo.Last().FileId;
                Telegram.Bot.Types.File fl = await client.GetFileAsync(photoId, cancellationToken: cancellationToken);
                var imageStream = new MemoryStream();
                await client.DownloadFileAsync(fl.FilePath, imageStream, cancellationToken: cancellationToken);
                var orig = new Bitmap(imageStream);
                orig.Save("original.jpg");
                // Если бы мы хотели получить битмап, то могли бы использовать new Bitmap(Image.FromStream(imageStream))
                // Но вместо этого пошлём картинку назад
                // Стрим помнит последнее место записи, мы же хотим теперь прочитать с самого начала
                var processed = processor.ProcessImage(orig);
                processed.Save("processed.jpg");

                var letter = recognise(processed);

                imageStream.Dispose();
                var outputStream = new MemoryStream();

                ImageCodecInfo jpgEncoder = ImageCodecInfo.GetImageEncoders().Single(x => x.FormatDescription == "JPEG");
                Encoder encoder2 = Encoder.Quality;
                EncoderParameters parameters = new EncoderParameters(1);
                EncoderParameter parameter = new EncoderParameter(encoder2, 50L);
                parameters.Param[0] = parameter;

                processed.Save(outputStream, jpgEncoder, parameters);
                
                var bytes = ((MemoryStream)outputStream).ToArray();
                Stream stream = new MemoryStream(bytes);

                var answer = aiml.Talk(chatId, username, "РАССКАЖИ О БУКВЕ " + char.ToUpper(letter));
                await client.SendPhotoAsync(
                    message.Chat.Id,
                    stream,
                    "Я тут вижу букву " + char.ToUpper(letter) + ".",
                    cancellationToken: cancellationToken
                );

                return;
            }
            // Можно обрабатывать разные виды сообщений, просто для примера пробросим реакцию на них в AIML
            if (message.Type == MessageType.Video)
            {
                await client.SendTextMessageAsync(message.Chat.Id, aiml.Talk(chatId, username, "Видео"), cancellationToken: cancellationToken);
                return;
            }
            if (message.Type == MessageType.Audio)
            {
                await client.SendTextMessageAsync(message.Chat.Id, aiml.Talk(chatId, username, "Аудио"), cancellationToken: cancellationToken);
                return;
            }
        }

        Task HandleErrorAsync(ITelegramBotClient botClient, Exception exception, CancellationToken cancellationToken)
        {
            var apiRequestException = exception as ApiRequestException;
            if (apiRequestException != null)
                Console.WriteLine($"Telegram API Error:\n[{apiRequestException.ErrorCode}]\n{apiRequestException.Message}");
            else
                Console.WriteLine(exception.ToString());
            return Task.CompletedTask;
        }

        public void Dispose()
        {
            // Заканчиваем работу - корректно отменяем задачи в других потоках
            // Отменяем токен - завершатся все асинхронные таски
            cts.Cancel();
        }

        private char recognise(Bitmap img)
        {
            Sample currentImage = new Sample(imgToData(img), symbolsCount);
            var recognizedClass = net.Predict(currentImage);
            return recognizedClass.ToString()[0];
        }

        private double[] imgToData(Bitmap img)
        {
            double[] res = new double[img.Width];

            for (int i = 0; i < img.Width; i++)
            {
                double sum = 0;
                for (int j = 0; j < img.Height; j++)
                {
                    var r = img.GetPixel(i, j).R;
                    sum += 1 - r * 1.0 / 255;
                }
                res[i] = sum;
            }
            return res;
        }
    }
}
