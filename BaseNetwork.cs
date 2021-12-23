using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AIMLTGBot
{
    //public enum FigureType : byte { А = 0, Б, В, Г, Д, Е, Ж, З, И, Й, К, Л, М, Н, О, П, Р, С, Т, У, Ф, Х, Ц, Ч, Ш, Щ, Ъ, Ы, Ь, Э, Ю, Я, Undef };
    public enum LetterType : byte { А = 0, Б, В, Г, Д, Е, Ж, З, И, К, Undef };
    /// <summary>
    /// Базовый класс для реализации как самодельного персептрона, так и обёртки для ActivationNetwork из Accord.Net
    /// </summary>
    abstract public class BaseNetwork
    {
        //  Делегат для информирования о процессе обучения (периодически извещает форму о том, сколько процентов работы сделано)
        //public FormUpdater updateDelegate = null;

        //Использовать double initialLearningRate = 0.25
        public abstract void ReInit(int[] structure, double initialLearningRate = 0.25);

        //Использовать bool parallel = true
        public abstract int Train(Sample sample, bool parallel = true);

        //Использовать bool parallel = true
        public abstract double TrainOnDataSet(SamplesSet samplesSet, int epochs_count, double acceptable_erorr, bool parallel = true);

        public abstract LetterType Predict(Sample sample, bool parallel = true);

        public abstract double TestOnDataSet(SamplesSet testSet);

        public abstract double[] getOutput();
    }
}
