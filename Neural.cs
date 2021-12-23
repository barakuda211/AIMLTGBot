using System;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.Collections;
using System.IO;
using Newtonsoft.Json;

namespace AIMLTGBot
{
    /// <summary>
    /// Класс для хранения образа – входной массив сигналов на сенсорах, выходные сигналы сети, и прочее
    /// </summary>
    public class Sample
    {
        /// <summary>
        /// Входной вектор
        /// </summary>
        public double[] input = null;

        /// <summary>
        /// Выходной вектор, задаётся извне как результат распознавания
        /// </summary>
        public double[] output = null;

        /// <summary>
        /// Вектор ошибки, вычисляется по какой-нибудь хитрой формуле
        /// </summary>
        public double[] error = null;

        /// <summary>
        /// Действительный класс образа. Указывается учителем
        /// </summary>
        public LetterType actualClass;

        /// <summary>
        /// Распознанный класс - определяется после обработки
        /// </summary>
        public LetterType recognizedClass;

        /// <summary>
        /// Конструктор образа - на основе входных данных для сенсоров, при этом можно указать класс образа, или не указывать
        /// </summary>
        /// <param name="inputValues"></param>
        /// <param name="sampleClass"></param>
        public Sample(double[] inputValues, int classesCount, LetterType sampleClass = LetterType.Undef)
        {
            //  Клонируем массивчик
            input = (double[])inputValues.Clone();
            output = new double[classesCount];
            if (sampleClass != LetterType.Undef) output[(int)sampleClass] = 1;


            recognizedClass = LetterType.Undef;
            actualClass = sampleClass;
        }

        /// <summary>
        /// Обработка реакции сети на данный образ на основе вектора выходов сети
        /// </summary>
        public void processOutput()
        {
            if (error == null)
                error = new double[output.Length];

            //  Нам так-то выход не нужен, нужна ошибка и определённый класс
            recognizedClass = 0;
            for (int i = 0; i < output.Length; ++i)
            {
                error[i] = ((i == (int)actualClass ? 1 : 0) - output[i]);
                if (output[i] > output[(int)recognizedClass]) recognizedClass = (LetterType)i;
            }
        }

        /// <summary>
        /// Вычисленная суммарная квадратичная ошибка сети. Предполагается, что целевые выходы - 1 для верного, и 0 для остальных
        /// </summary>
        /// <returns></returns>
        public double EstimatedError()
        {
            double Result = 0;
            for (int i = 0; i < output.Length; ++i)
                Result += System.Math.Pow(error[i], 2);
            return Result;
        }

        /// <summary>
        /// Добавляет к аргументу ошибку, соответствующую данному образу (не квадратичную!!!)
        /// </summary>
        /// <param name="errorVector"></param>
        /// <returns></returns>
        public void updateErrorVector(double[] errorVector)
        {
            for (int i = 0; i < errorVector.Length; ++i)
                errorVector[i] += error[i];
        }

        /// <summary>
        /// Представление в виде строки
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            string result = "Sample decoding : " + actualClass.ToString() + "(" + ((int)actualClass).ToString() + "); " + Environment.NewLine + "Input : ";
            for (int i = 0; i < input.Length; ++i) result += input[i].ToString() + "; ";
            result += Environment.NewLine + "Output : ";
            if (output == null) result += "null;";
            else
                for (int i = 0; i < output.Length; ++i) result += output[i].ToString() + "; ";
            result += Environment.NewLine + "Error : ";

            if (error == null) result += "null;";
            else
                for (int i = 0; i < error.Length; ++i) result += error[i].ToString() + "; ";
            result += Environment.NewLine + "Recognized : " + recognizedClass.ToString() + "(" + ((int)recognizedClass).ToString() + "); " + Environment.NewLine;


            return result;
        }

        /// <summary>
        /// Правильно ли распознан образ
        /// </summary>
        /// <returns></returns>
        public bool Correct() { return actualClass == recognizedClass; }
    }

    /// <summary>
    /// Выборка образов. Могут быть как классифицированные (обучающая, тестовая выборки), так и не классифицированные (обработка)
    /// </summary>
    public class SamplesSet : IEnumerable
    {
        /// <summary>
        /// Накопленные обучающие образы
        /// </summary>
        public List<Sample> samples = new List<Sample>();

        private Random rnd = new Random();

        /// <summary>
        /// Добавление образа к коллекции
        /// </summary>
        /// <param name="image"></param>
        public void AddSample(Sample image)
        {
            samples.Add(image);
        }
        public int Count { get { return samples.Count; } }

        public IEnumerator GetEnumerator()
        {
            return samples.GetEnumerator();
        }

        public void Randomize()
        {
            for (int i = 0; i < samples.Count; i++)
            {
                var temp = samples[i];
                var rand = rnd.Next(0, samples.Count);
                samples[i] = samples[rand];
                samples[rand] = temp;
            }
        }

        /// <summary>
        /// Реализация доступа по индексу
        /// </summary>
        /// <param name="i"></param>
        /// <returns></returns>
        public Sample this[int i]
        {
            get { return samples[i]; }
            set { samples[i] = value; }
        }

        public double ErrorsCount()
        {
            double correct = 0;
            double wrong = 0;
            foreach (var sample in samples)
                if (sample.Correct()) ++correct; else ++wrong;
            return correct / (correct + wrong);
        }
    }


    public class Neuron
    {

        public double OutSignal = 0;
        public double Error = 0;
        public double Bias = -1;
        public double BiasWeight = 0.01;
        public static Random rnd = new Random();
        public int CountInput = 0;
        public double[] Weights;
        public Neuron[] InputLayer;

        /// <summary>
        /// Инициализация весов случайными значениями на отрезке [min; max]
        /// </summary>
        private void InitWeight(int cnt, double min, double max)
        {
            Weights = new double[cnt];
            for (int i = 0; i < cnt; ++i)
                Weights[i] = min + rnd.NextDouble() * (max - min);
        }
        /// <summary>
        /// Конструктор нейрона от предыдущего слоя
        /// </summary>
        public Neuron(Neuron[] previous)
        {
            if (previous == null)
                return;
            InputLayer = previous;
            CountInput = InputLayer.Length;
            InitWeight(CountInput, -1, 1);
        }

        public Neuron(NeuronJson nj,Neuron[] previous)
        {
            if (nj.weight != null)
            {
                Weights = new double[nj.weight.Length];
                Array.Copy(nj.weight, Weights, nj.weight.Length);
            }
            if (previous != null)
            {
                InputLayer = previous;
                CountInput = InputLayer.Length;
            }
        }

        public Neuron() { }

        /// <summary>
        /// Применение функции активации
        /// </summary>
        public void Activation()
        {
            OutSignal = ActivationFunction(BiasWeight * Bias + InputLayer.Zip(Weights, (neuron, w) => neuron.OutSignal * w).Sum());
        }

        /// <summary>
        /// Пересчет весов на заданном фрагменте слоя
        /// </summary>
        private void RecalcWeights(double alpha, int from, int to)
        {
            for (int i = from; i < to; ++i)
                Weights[i] += alpha * InputLayer[i].OutSignal * Error;
        }
        /// <summary>
        /// Вспомогательная функция, вызываемая перед распространением ошибки
        /// </summary>
        public void HelpBackPropagation(double alpha)
        {
            Error *= OutSignal * (1 - OutSignal);
            BiasWeight += alpha * Bias * Error;
        }

        /// <summary>
        /// Сброс ошибки
        /// </summary>
        public void ResetError() => Error = 0;

        /// <summary>
        /// Распространение ошибки на предыдущий слой и пересчет весов
        /// </summary>
        public void BackPropagation(double alpha)
        {
            HelpBackPropagation(alpha);

            for (int i = 0; i < CountInput; ++i)
                InputLayer[i].Error += Error * Weights[i];

            RecalcWeights(alpha, 0, CountInput);
            ResetError();
        }


        /// <summary>
        /// Распространение ошибки на фрагмент предыдущего слоя и пересчет весов
        /// </summary>
        public void BackPropagationParallel(double alpha, int from, int to)
        {
            for (int i = from; i < to; ++i)
                InputLayer[i].Error += Error * Weights[i];
            RecalcWeights(alpha, from, to);
        }
        /// <summary>
        /// Функция активации (сигмоид)
        /// </summary>
        public static double ActivationFunction(double x) => 1 / (1 + System.Math.Exp(-x));
    }

    public class NeuronJson
    {
        public double[] weight;

        [JsonConstructor]
        public NeuronJson(double[] n)
        {
            weight = n;
        }

        public NeuronJson(Neuron n)
        {
            if (n.Weights != null)
            {
                weight = new double[n.Weights.Length];
                Array.Copy(n.Weights, weight, n.Weights.Length);
            }
        }
    }


    class NetJson
    {
        public int CntLayers;
        public int CntSensors;
        public int CntClasses;
        public NeuronJson[][] layers;

        [JsonConstructor]
        public NetJson(int CntLayers, int CntSensors, int CntClasses, NeuronJson[][] layers)
        {
            this.CntLayers = CntLayers;
            this.CntSensors = CntSensors;
            this.CntClasses = CntClasses;
            this.layers = layers;
            
        }

        public NetJson(NeuralNetwork nn)
        {
            CntLayers = nn.CntLayers;
            CntSensors = nn.CntSensors;
            CntClasses = nn.CntClasses;
            layers = new NeuronJson[nn.Layers.Length][];
            for (int i = 0; i < nn.Layers.Length; i++)
            {
                layers[i] = new NeuronJson[nn.Layers[i].Length];
                for (int j = 0; j < layers[i].Length; j++)
                {
                    
                        layers[i][j] = new NeuronJson(nn.Layers[i][j]);
                }
                    //Array.Copy(nn.Layers[i][j].Weights, layers[i][j].weight, nn.Layers[i][j].Weights.Length);
            }
        }
    }

    public class NeuralNetwork : BaseNetwork
    {
        /// <summary>
        /// Скорость обучения
        /// </summary>
        public double alpha = 0.25;
        /// <summary>
        /// Все слои
        /// </summary>
        public Neuron[][] Layers;

        public Stopwatch stopWatch = new Stopwatch();
        /// <summary>
        /// Число слоев
        /// </summary>
        public int CntLayers;
        /// <summary>
        /// Число сенсоров
        /// </summary>
        public int CntSensors;
        /// <summary>
        /// Число классов (число нейронов на последнем слое)
        /// </summary>
        public int CntClasses;

        public NeuralNetwork(string filename = "net.json")
        {
            LoadNetwork(filename);
        }

        public NeuralNetwork(int[] structure)
        {
            CntLayers = structure.Length;
            CntClasses = structure.Last();
            CntSensors = structure[0];
            ReInit(structure);
        }

        private void InitFromAnotherNetwork(NetJson nj)
        {
            CntLayers = nj.CntLayers;
            CntClasses = nj.CntClasses;
            CntSensors = nj.CntSensors;
            if (Layers == null)
            {
                Layers = new Neuron[nj.layers.Length][];
                for (int i = 0; i < nj.layers.Length; i++)
                    Layers[i] = new Neuron[nj.layers[i].Length];
            }
            for (int i =0; i <nj.layers.Length;i++)
                for (int j = 0; j < nj.layers[i].Length; j++)
                {
                    if (i > 0)
                        Layers[i][j] = new Neuron(nj.layers[i][j], Layers[i - 1]);
                    else
                        Layers[i][j] = new Neuron(nj.layers[i][j], null);
                }
        }

        private void InitFromAnotherNetwork(NeuralNetwork nn)
        {
            alpha = nn.alpha;
            Layers = nn.Layers;
            CntLayers = nn.CntLayers;
            CntSensors = nn.CntSensors;
            CntClasses = nn.CntClasses;
        }

        public NeuralNetwork(NeuralNetwork nn)
        {
            InitFromAnotherNetwork(nn);
        }

        public override void ReInit(int[] structure, double initialLearningRate = 0.25)
        {
            if (structure.Length < 2)
                throw new Exception("Недостаточно слоев");

            alpha = initialLearningRate;

            // Заводим массив слоев
            Layers = new Neuron[CntLayers][];

            // Сенсорный слой, у нейронов пустой вход
            Layers[0] = new Neuron[CntSensors];
            for (int j = 0; j < CntSensors; ++j)
                Layers[0][j] = new Neuron();

            // Все последующие слои. Каждый нейрон имеет ссылку на предыдущий слой
            for (int nLayer = 1; nLayer < CntLayers; ++nLayer)
            {
                Layers[nLayer] = new Neuron[structure[nLayer]];
                for (int j = 0; j < structure[nLayer]; ++j)
                    Layers[nLayer][j] = new Neuron(Layers[nLayer - 1]);
            }
        }
        /// <summary>
        /// Вспомогательная функция, запускающая обучение
        /// </summary>
        private void HelpRun(Sample sample, bool parallel = true)
        {
            if (parallel)
                RunParallel(sample);
            else
                Run(sample);
        }
        /// <summary>
        /// Непараллельное обучение
        /// </summary>
        private void Run(Sample image)
        {
            if (image.input.Length != Layers[0].Length)
                throw new Exception("Неверно задано число входных сенсоров");

            // Заполняем сенсоры
            for (int i = 0; i < image.input.Length; ++i)
                Layers[0][i].OutSignal = image.input[i];

            // Выполняем активацию по всем слоям, кроме сенсорного
            for (int i = 1; i < CntLayers; ++i)
                for (int j = 0; j < Layers[i].Length; ++j)
                    Layers[i][j].Activation();
            // Сохраняем выходной сигнал
            for (int i = 0; i < CntClasses; ++i)
                image.output[i] = Layers[CntLayers - 1][i].OutSignal;

            image.processOutput();
        }

        /// <summary>
        /// Параллельное обучение
        /// </summary>
        private void RunParallel(Sample image)
        {
            if (image.input.Length != Layers[0].Length)
                throw new Exception("Неверно задано число входных сенсоров");

            // Заполняем сенсоры
            for (int i = 0; i < image.input.Length; ++i)
                Layers[0][i].OutSignal = image.input[i];

            // Выполняем параллельную активацию
            for (int i = 1; i < CntLayers; ++i)
            {
                Parallel.For(0, Layers[i].Length, j =>
                {
                    Layers[i][j].Activation();
                });
            }
            // Сохраняем выходной сигнал
            for (int i = 0; i < CntClasses; ++i)
                image.output[i] = Layers[CntLayers - 1][i].OutSignal;

            image.processOutput();
        }

        /// <summary>
        /// Запускает обратное распространение ошибки
        /// </summary>
        private void HelpBackPropagation(Sample image, bool parallel = true)
        {
            if (parallel)
                BackPropagationParallel(image);
            else
                BackPropagation(image);
        }

        /// <summary>
        /// Распространяет ошибку обратно и пересчитывает веса
        /// </summary>
        private void BackPropagation(Sample image)
        {
            // Считываем ошибку из образа на входной слой
            for (int i = 0; i < CntClasses; i++)
                Layers[Layers.Length - 1][i].Error = image.error[i];
            // Вызываем функцию обратного распространения ошибки для каждого нейрона каждого слоя
            for (int i = CntLayers - 1; i >= 0; --i)
                for (int j = 0; j < Layers[i].Length; ++j)
                    Layers[i][j].BackPropagation(alpha);
        }

        /// <summary>
        /// Параллельное обратное распространение ошибки
        /// </summary>
        private void BackPropagationParallel(Sample image)
        {
            // Число потоков, каждый из которых обрабатывает один фрагмент предыдущего слоя
            int cntThreads = 16;

            // Считываем ошибку из образа и записываем во входной слой
            for (int i = 0; i < CntClasses; i++)
                Layers[CntLayers - 1][i].Error = image.error[i];

            for (int layer = CntLayers - 1; layer > 0; --layer)
            {
                int len = Layers[layer - 1].Length / cntThreads;

                // Сбрасываем ошибку на предыдущем слое
                for (int j = 0; j < Layers[layer - 1].Length; ++j)
                    Layers[layer - 1][j].ResetError();
                // Выполняем обратное распространение
                for (int j = 0; j < Layers[layer].Length; ++j)
                    Layers[layer][j].HelpBackPropagation(alpha);
                Parallel.For(0, cntThreads, i =>
                {
                    for (int j = 0; j < Layers[layer].Length; ++j)
                        Layers[layer][j].BackPropagationParallel(alpha, len * i, i == cntThreads ? Layers[layer - 1].Length : len * (i + 1));
                });

            }
        }
        /// <summary>
        /// Распознавание заданного образа
        /// </summary>
        public override LetterType Predict(Sample sample, bool parallel = true)
        {
            Run(sample);
            return sample.recognizedClass;
        }

        /// <summary>
        /// Обучение заданному образу
        /// </summary>
        public override int Train(Sample sample, bool parallel = true)
        {
            int cntIters = 0;

            do
            {
                HelpRun(sample, parallel);

                if (sample.Correct() && sample.EstimatedError() < 0.1)
                    return cntIters;

                HelpBackPropagation(sample, parallel);

            } while (cntIters++ < 100);

            return cntIters;
        }

        public override double[] getOutput() => Layers.Last().Select(n => n.OutSignal).ToArray();

        /// <summary>
        /// Обучение на всем датасете
        /// </summary>
        public override double TrainOnDataSet(SamplesSet samplesSet, int epochs_count, double acceptable_erorr, bool parallel = true)
        {
            double cntTrue = 0;
            stopWatch.Restart();
            samplesSet.Randomize();
            for (int nEpoch = 0; nEpoch < epochs_count; ++nEpoch)
            {
                cntTrue = 0;
                for (int i = 0; i < samplesSet.samples.Count; ++i)
                {
                    if (Train(samplesSet.samples[i], parallel) == 0)
                        cntTrue++;
                    //updateDelegate?.Invoke((double)i / samplesSet.samples.Count, cntTrue / (i + 1), stopWatch.Elapsed, nEpoch + 1);
                }

                if (1 - cntTrue / samplesSet.samples.Count <= acceptable_erorr)
                    break;
            }

            stopWatch.Stop();
            //updateDelegate?.Invoke(1, cntTrue / samplesSet.samples.Count, stopWatch.Elapsed);

            return cntTrue / samplesSet.samples.Count *100.0;
        }
        /// <summary>
        /// Проверка на тестовом множестве
        /// </summary>
        public override double TestOnDataSet(SamplesSet testSet)
        {
            int cntTrue = 0;
            for (int i = 0; i < testSet.Count; ++i)
            {
                Sample s = testSet.samples[i];
                Predict(s);
                if (s.Correct())
                    cntTrue++;
            }
            return (double)cntTrue / testSet.Count;
        }

        public void SaveNetwork(string filename = "net.json")
        {
            string net_js = JsonConvert.SerializeObject(new NetJson(this), Formatting.Indented);
            File.WriteAllText(filename, net_js);
        }

        public void LoadNetwork(string filename = "net.json")
        {
            InitFromAnotherNetwork(JsonConvert.DeserializeObject<NetJson>(File.ReadAllText(filename)));
        }
    }
}

