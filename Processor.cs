using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using AForge;

namespace AIMLTGBot
{
    public class Settings
    {
        private int _border = 20;
        public int border
        {
            get
            {
                return _border;
            }
            set
            {
                if ((value > 0) && (value < height / 3))
                {
                    _border = value;
                    if (top > 2 * _border) top = 2 * _border;
                    if (left > 2 * _border) left = 2 * _border;
                }
            }
        }

        public int width = 640;
        public int height = 640;
        

        /// <summary>
        /// Желаемый размер изображения до обработки
        /// </summary>
        public Size orignalDesiredSize = new Size(500, 500);
        /// <summary>
        /// Желаемый размер изображения после обработки
        /// </summary>
        public Size processedDesiredSize = new Size(500, 500);

        public int margin = 10;
        public int top = 40;
        public int left = 40;

        /// <summary>
        /// Второй этап обработки
        /// </summary>
        public bool processImg = false;

        /// <summary>
        /// Порог при отсечении по цвету 
        /// </summary>
        public byte threshold = 120;
        public float differenceLim = 0.15f;
    }

    public class MagicEye
    {
        /// <summary>
        /// Обработанное изображение
        /// </summary>
        public Bitmap processed;

        /// <summary>
        /// Класс настроек
        /// </summary>
        public Settings settings = new Settings();

        public MagicEye() { }

        public Bitmap ProcessImage(Bitmap bitmap)
        {
            //поиск центра картинки
            int w_space = 0, h_space = 0;
            if (bitmap.Width > bitmap.Height)
                w_space = (bitmap.Width - bitmap.Height) / 2;
            else
                h_space = (bitmap.Height - bitmap.Width) / 2;

            //  Теперь всю эту муть пилим в обработанное изображение
            AForge.Imaging.Filters.Grayscale grayFilter = new AForge.Imaging.Filters.Grayscale(0.2125, 0.7154, 0.0721);
            var uProcessed = grayFilter.Apply(AForge.Imaging.UnmanagedImage.FromManagedImage(bitmap));

            //AForge.Imaging.Filters.ResizeBilinear scaleFilter = new AForge.Imaging.Filters.ResizeBilinear(Math.Min(bitmap.Width, bitmap.Height), Math.Min(bitmap.Width, bitmap.Height));
            AForge.Imaging.Filters.Crop scaleFilter = new AForge.Imaging.Filters.Crop(new Rectangle(w_space,h_space,Math.Min(bitmap.Width, bitmap.Height), Math.Min(bitmap.Width, bitmap.Height)));
            uProcessed = scaleFilter.Apply(uProcessed);

            //  Пороговый фильтр применяем. Величина порога берётся из настроек, и меняется на форме
            //AForge.Imaging.Filters.Threshold threshldFilter = new AForge.Imaging.Filters.Threshold();
            //threshldFilter.ThresholdValue = settings.threshold;
            //threshldFilter.ApplyInPlace(uProcessed);
            AForge.Imaging.Filters.BradleyLocalThresholding threshldFilter = new AForge.Imaging.Filters.BradleyLocalThresholding();
            threshldFilter.PixelBrightnessDifferenceLimit = settings.differenceLim;
            threshldFilter.ApplyInPlace(uProcessed);

            processSample(ref uProcessed);

            processed = uProcessed.ToManagedImage();
            return processed;
        }

        /// <summary>
        /// Обработка одного сэмпла
        /// </summary>
        /// <param name="index"></param>
        private void processSample(ref AForge.Imaging.UnmanagedImage unmanaged)
        {

            // Обрезаем края, оставляя только центральные блобчики
            //AForge.Imaging.Filters.Crop cropFilter = new AForge.Imaging.Filters.Crop(new Rectangle(lx, ly, rx - lx + 10, ry - ly + 10));
            //unmanaged = cropFilter.Apply(unmanaged);

            //  Масштабируем до 100x100
            AForge.Imaging.Filters.ResizeBilinear scaleFilter = new AForge.Imaging.Filters.ResizeBilinear(100, 100);
            unmanaged = scaleFilter.Apply(unmanaged);
        }
    }
}

