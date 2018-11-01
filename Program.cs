using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;

namespace Redes_Bayesianas
{
    class Autism
    {
        static void Main(string[] args)
        {
            //Probabilities of each variable

            //S
            Variable<bool> S = Variable.Bernoulli(0.5);
            //Male = false; Female = true;

            //F35
            Variable<bool> F35 = Variable.Bernoulli(0.3784);

            //M35
            Variable<bool> M35 = Variable.Bernoulli(0.2225);

            //OSA
            Variable<bool> OSA = Variable.New<bool>();
            using (Variable.If(F35))
            {
                using (Variable.If(M35))
                {
                    using (Variable.If(S)) OSA.SetTo(Variable.Bernoulli(0.0081));     //y;y;f
                    using (Variable.IfNot(S)) OSA.SetTo(Variable.Bernoulli(0.0316));  //y;y;m
                }
                using (Variable.IfNot(M35))
                {
                    using (Variable.If(S)) OSA.SetTo(Variable.Bernoulli(0.0066));     //y;n;f
                    using (Variable.IfNot(S)) OSA.SetTo(Variable.Bernoulli(0.0259));  //y;n;m
                }
            }
            using (Variable.IfNot(F35))
            {
                using (Variable.If(M35))
                {
                    using (Variable.If(S)) OSA.SetTo(Variable.Bernoulli(0.0065));     //n;y;f
                    using (Variable.IfNot(S)) OSA.SetTo(Variable.Bernoulli(0.0253));  //n;y;m
                }
                using (Variable.IfNot(M35))
                {
                    using (Variable.If(S)) OSA.SetTo(Variable.Bernoulli(0.005));      //n;n;f
                    using (Variable.IfNot(S)) OSA.SetTo(Variable.Bernoulli(0.0196));  //n;n;m
                }
            }

            //FYSA
            Variable<bool> FYSA = Variable.New<bool>();
            using (Variable.If(F35))
            {
                using (Variable.If(M35))
                {
                    using (Variable.If(S))
                    {
                        using (Variable.If(OSA)) FYSA.SetTo(Variable.Bernoulli(0.1224));       //y;y;f;y
                        using (Variable.IfNot(OSA)) FYSA.SetTo(Variable.Bernoulli(0.0081));    //y;y;f;n
                    }
                    using (Variable.IfNot(S))
                    {
                        using (Variable.If(OSA)) FYSA.SetTo(Variable.Bernoulli(0.0676));       //y;y;m;y
                        using (Variable.IfNot(OSA)) FYSA.SetTo(Variable.Bernoulli(0.0081));    //y;y;m;n
                    }
                }
                using (Variable.IfNot(M35))
                {
                    using (Variable.If(S))
                    {
                        using (Variable.If(OSA)) FYSA.SetTo(Variable.Bernoulli(0.1003));       //y;n;f;y
                        using (Variable.IfNot(OSA)) FYSA.SetTo(Variable.Bernoulli(0.0066));    //y;n;f;n
                    }
                    using (Variable.IfNot(S))
                    {
                        using (Variable.If(OSA)) FYSA.SetTo(Variable.Bernoulli(0.0554));       //y;n;m;y
                        using (Variable.IfNot(OSA)) FYSA.SetTo(Variable.Bernoulli(0.0066));    //y;n;m;n
                    }
                }
            }
            using (Variable.IfNot(F35))
            {
                using (Variable.If(M35))
                {
                    using (Variable.If(S))
                    {
                        using (Variable.If(OSA)) FYSA.SetTo(Variable.Bernoulli(0.098));        //n;y;f;y
                        using (Variable.IfNot(OSA)) FYSA.SetTo(Variable.Bernoulli(0.0065));    //n;y;f;n
                    }
                    using (Variable.IfNot(S))
                    {
                        using (Variable.If(OSA)) FYSA.SetTo(Variable.Bernoulli(0.0542));       //n;y;m;y
                        using (Variable.IfNot(OSA)) FYSA.SetTo(Variable.Bernoulli(0.0065));    //n;y;m;n
                    }
                }
                using (Variable.IfNot(M35))
                {
                    using (Variable.If(S))
                    {
                        using (Variable.If(OSA)) FYSA.SetTo(Variable.Bernoulli(0.076));        //n;n;f;y
                        using (Variable.IfNot(OSA)) FYSA.SetTo(Variable.Bernoulli(0.005));     //n;n;f;n
                    }
                    using (Variable.IfNot(S))
                    {
                        using (Variable.If(OSA)) FYSA.SetTo(Variable.Bernoulli(0.042));        //n;n;m;y
                        using (Variable.IfNot(OSA)) FYSA.SetTo(Variable.Bernoulli(0.005));     //n;n;m;n
                    }
                }
            }

            //MYSA
            Variable<bool> MYSA = Variable.New<bool>();
            using (Variable.If(F35))
            {
                using (Variable.If(M35))
                {
                    using (Variable.If(S))
                    {
                        using (Variable.If(OSA)) MYSA.SetTo(Variable.Bernoulli(0.2689));       //y;y;f;y
                        using (Variable.IfNot(OSA)) MYSA.SetTo(Variable.Bernoulli(0.0316));    //y;y;f;n
                    }
                    using (Variable.IfNot(S))
                    {
                        using (Variable.If(OSA)) MYSA.SetTo(Variable.Bernoulli(0.2077));       //y;y;m;y
                        using (Variable.IfNot(OSA)) MYSA.SetTo(Variable.Bernoulli(0.0316));    //y;y;m;n
                    }
                }
                using (Variable.IfNot(M35))
                {
                    using (Variable.If(S))
                    {
                        using (Variable.If(OSA)) MYSA.SetTo(Variable.Bernoulli(0.2204));       //y;n;f;y
                        using (Variable.IfNot(OSA)) MYSA.SetTo(Variable.Bernoulli(0.0259));    //y;n;f;n
                    }
                    using (Variable.IfNot(S))
                    {
                        using (Variable.If(OSA)) MYSA.SetTo(Variable.Bernoulli(0.1703));       //y;n;m;y
                        using (Variable.IfNot(OSA)) MYSA.SetTo(Variable.Bernoulli(0.0259));    //y;n;m;n
                    }
                }
            }
            using (Variable.IfNot(F35))
            {
                using (Variable.If(M35))
                {
                    using (Variable.If(S))
                    {
                        using (Variable.If(OSA)) MYSA.SetTo(Variable.Bernoulli(0.2154));       //n;y;f;y
                        using (Variable.IfNot(OSA)) MYSA.SetTo(Variable.Bernoulli(0.0253));    //n;y;f;n
                    }
                    using (Variable.IfNot(S))
                    {
                        using (Variable.If(OSA)) MYSA.SetTo(Variable.Bernoulli(0.1664));       //n;y;m;y
                        using (Variable.IfNot(OSA)) MYSA.SetTo(Variable.Bernoulli(0.0253));    //n;y;m;n
                    }
                }
                using (Variable.IfNot(M35))
                {
                    using (Variable.If(S))
                    {
                        using (Variable.If(OSA)) MYSA.SetTo(Variable.Bernoulli(0.167));        //n;n;f;y
                        using (Variable.IfNot(OSA)) MYSA.SetTo(Variable.Bernoulli(0.0196));    //n;n;f;n
                    }
                    using (Variable.IfNot(S))
                    {
                        using (Variable.If(OSA)) MYSA.SetTo(Variable.Bernoulli(0.129));        //n;n;m;y
                        using (Variable.IfNot(OSA)) MYSA.SetTo(Variable.Bernoulli(0.0196));    //n;n;m;n
                    }
                }
            }

            //MA
            Variable<bool> MA = Variable.New<bool>();
            using (Variable.If(OSA)) MA.SetTo(Variable.Bernoulli(0.166));
            using (Variable.IfNot(OSA)) MA.SetTo(Variable.Bernoulli(0.005));

            //FA
            Variable<bool> FA = Variable.New<bool>();
            using (Variable.If(OSA)) FA.SetTo(Variable.Bernoulli(0.664));
            using (Variable.IfNot(OSA)) FA.SetTo(Variable.Bernoulli(0.0196));

            //GMA1
            Variable<bool> GMA1 = Variable.New<bool>();
            using (Variable.If(FA)) GMA1.SetTo(Variable.Bernoulli(0.166));
            using (Variable.IfNot(FA)) GMA1.SetTo(Variable.Bernoulli(0.005));

            //GFA1
            Variable<bool> GFA1 = Variable.New<bool>();
            using (Variable.If(FA)) GFA1.SetTo(Variable.Bernoulli(0.664));
            using (Variable.IfNot(FA)) GFA1.SetTo(Variable.Bernoulli(0.0196));

            //GMA2
            Variable<bool> GMA2 = Variable.New<bool>();
            using (Variable.If(MA)) GMA2.SetTo(Variable.Bernoulli(0.166));
            using (Variable.IfNot(MA)) GMA2.SetTo(Variable.Bernoulli(0.005));

            //GFA2
            Variable<bool> GFA2 = Variable.New<bool>();
            using (Variable.If(MA)) GFA2.SetTo(Variable.Bernoulli(0.664));
            using (Variable.IfNot(MA)) GFA2.SetTo(Variable.Bernoulli(0.0196));

            //Inference

            InferenceEngine ie = new InferenceEngine();
            ie.ShowProgress = false;

            //inicial test of the variables
            Console.WriteLine();
            Console.WriteLine("  P(OSA)  = " + ie.Infer(OSA));
            Console.WriteLine("  P(MYSA) = " + ie.Infer(MYSA));
            Console.WriteLine("  P(FYSA) = " + ie.Infer(FYSA));
            Console.WriteLine("  P(MA)   = " + ie.Infer(MA));
            Console.WriteLine("  P(FA)   = " + ie.Infer(FA));
            Console.WriteLine("  P(GMA1) = " + ie.Infer(GMA1));
            Console.WriteLine("  P(GFA1) = " + ie.Infer(GFA1));
            Console.WriteLine("  P(GMA2) = " + ie.Infer(GMA2));
            Console.WriteLine("  P(GFA2) = " + ie.Infer(GFA2));

            //male older sibling with autism
            S.ObservedValue = false;
            OSA.ObservedValue = true;
            Console.WriteLine();
            Console.WriteLine("  P(MYSA | ¬S, OSA) = " + ie.Infer(MYSA));
            Console.WriteLine("  P(FYSA | ¬S, OSA) = " + ie.Infer(FYSA));
            Console.WriteLine("  P(MA   | ¬S, OSA) = " + ie.Infer(MA));
            Console.WriteLine("  P(FA   | ¬S, OSA) = " + ie.Infer(FA));
            Console.WriteLine("  P(GMA1 | ¬S, OSA) = " + ie.Infer(GMA1));
            Console.WriteLine("  P(GFA1 | ¬S, OSA) = " + ie.Infer(GFA1));
            Console.WriteLine("  P(GMA2 | ¬S, OSA) = " + ie.Infer(GMA2));
            Console.WriteLine("  P(GFA2 | ¬S, OSA) = " + ie.Infer(GFA2));

            //female older sibling with autism
            S.ObservedValue = true;
            OSA.ObservedValue = true;
            Console.WriteLine();
            Console.WriteLine("  P(MYSA | S, OSA) = " + ie.Infer(MYSA));
            Console.WriteLine("  P(FYSA | S, OSA) = " + ie.Infer(FYSA));
            Console.WriteLine("  P(MA   | S, OSA) = " + ie.Infer(MA));
            Console.WriteLine("  P(FA   | S, OSA) = " + ie.Infer(FA));
            Console.WriteLine("  P(GMA1 | S, OSA) = " + ie.Infer(GMA1));
            Console.WriteLine("  P(GFA1 | S, OSA) = " + ie.Infer(GFA1));
            Console.WriteLine("  P(GMA2 | S, OSA) = " + ie.Infer(GMA2));
            Console.WriteLine("  P(GFA2 | S, OSA) = " + ie.Infer(GFA2));

            //father's age >= 35 and mother's age < 35
            S.ClearObservedValue();
            OSA.ClearObservedValue();
            F35.ObservedValue = true;
            M35.ObservedValue = false;
            Console.WriteLine();
            Console.WriteLine("  P(OSA  | F35, ¬M35) = " + ie.Infer(OSA));
            Console.WriteLine("  P(MYSA | F35, ¬M35) = " + ie.Infer(MYSA));
            Console.WriteLine("  P(FYSA | F35, ¬M35) = " + ie.Infer(FYSA));

            //father's age < 35 and mother's age >= 35
            F35.ObservedValue = false;
            M35.ObservedValue = true;
            Console.WriteLine();
            Console.WriteLine("  P(OSA  | ¬F35, M35) = " + ie.Infer(OSA));
            Console.WriteLine("  P(MYSA | ¬F35, M35) = " + ie.Infer(MYSA));
            Console.WriteLine("  P(FYSA | ¬F35, M35) = " + ie.Infer(FYSA));

            //father's age >= 35 and mother's age >= 35
            F35.ObservedValue = true;
            M35.ObservedValue = true;
            Console.WriteLine();
            Console.WriteLine("  P(OSA  | F35, M35) = " + ie.Infer(OSA));
            Console.WriteLine("  P(MYSA | F35, M35) = " + ie.Infer(MYSA));
            Console.WriteLine("  P(FYSA | F35, M35) = " + ie.Infer(FYSA));

            //everithing related to OSA (male) = false except for FA = true
            S.ObservedValue = false;
            F35.ObservedValue = false;
            M35.ObservedValue = false;
            MYSA.ObservedValue = false;
            FYSA.ObservedValue = false;
            MA.ObservedValue = false;
            FA.ObservedValue = true;
            Console.WriteLine();
            Console.WriteLine("  P(OSA | ¬S, ¬F35, ¬M35, ¬MYSA, ¬FYSA, ¬MA, FA) = " + ie.Infer(OSA));

            //everithing related to OSA (male) = false except for MA = true
            S.ObservedValue = false;
            F35.ObservedValue = false;
            M35.ObservedValue = false;
            MYSA.ObservedValue = false;
            FYSA.ObservedValue = false;
            MA.ObservedValue = true;
            FA.ObservedValue = false;
            Console.WriteLine();
            Console.WriteLine("  P(OSA | ¬S, ¬F35, ¬M35, ¬MYSA, ¬FYSA, MA, ¬FA) = " + ie.Infer(OSA));

            //everithing related to OSA = false except for FA = true and S = true (female)
            S.ObservedValue = true;
            F35.ObservedValue = false;
            M35.ObservedValue = false;
            MYSA.ObservedValue = false;
            FYSA.ObservedValue = false;
            MA.ObservedValue = false;
            FA.ObservedValue = true;
            Console.WriteLine();
            Console.WriteLine("  P(OSA | S, ¬F35, ¬M35, ¬MYSA, ¬FYSA, ¬MA, FA) = " + ie.Infer(OSA));

            //everithing related to OSA = false except for MA = true and S = true (female)
            S.ObservedValue = true;
            F35.ObservedValue = false;
            M35.ObservedValue = false;
            MYSA.ObservedValue = false;
            FYSA.ObservedValue = false;
            MA.ObservedValue = true;
            FA.ObservedValue = false;
            Console.WriteLine();
            Console.WriteLine("  P(OSA | S, ¬F35, ¬M35, ¬MYSA, ¬FYSA, MA, ¬FA) = " + ie.Infer(OSA));

            //everithing related to OSA (male) = false except for FA, MA = true
            S.ObservedValue = false;
            F35.ObservedValue = false;
            M35.ObservedValue = false;
            MYSA.ObservedValue = false;
            FYSA.ObservedValue = false;
            MA.ObservedValue = true;
            FA.ObservedValue = true;
            Console.WriteLine();
            Console.WriteLine("  P(OSA | ¬S, ¬F35, ¬M35, ¬MYSA, ¬FYSA, MA, FA) = " + ie.Infer(OSA));

            //everithing related to OSA = false except for FA, MA = true and S = true (female)
            S.ObservedValue = true;
            F35.ObservedValue = false;
            M35.ObservedValue = false;
            MYSA.ObservedValue = false;
            FYSA.ObservedValue = false;
            MA.ObservedValue = true;
            FA.ObservedValue = true;
            Console.WriteLine();
            Console.WriteLine("  P(OSA | S, ¬F35, ¬M35, ¬MYSA, ¬FYSA, MA, FA) = " + ie.Infer(OSA));

            //GFA1 = true, OSA (male)
            S.ObservedValue = false;
            F35.ObservedValue = false;
            M35.ObservedValue = false;
            MYSA.ObservedValue = false;
            FYSA.ObservedValue = false;
            MA.ObservedValue = false;
            FA.ClearObservedValue();
            GFA1.ObservedValue = true;
            GMA1.ObservedValue = false;
            Console.WriteLine();
            Console.WriteLine("  P(OSA | ¬S, ¬F35, ¬M35, ¬MYSA, ¬FYSA, ¬MA, GFA1, ¬GMA1) = " + ie.Infer(OSA));

            //GFA1 = true, OSA (female)
            S.ObservedValue = true;
            F35.ObservedValue = false;
            M35.ObservedValue = false;
            MYSA.ObservedValue = false;
            FYSA.ObservedValue = false;
            MA.ObservedValue = false;
            FA.ClearObservedValue();
            GFA1.ObservedValue = true;
            GMA1.ObservedValue = false;
            Console.WriteLine();
            Console.WriteLine("  P(OSA | S, ¬F35, ¬M35, ¬MYSA, ¬FYSA, ¬MA, GFA1, ¬GMA1) = " + ie.Infer(OSA));

            //GMA1 = true, OSA (male)
            S.ObservedValue = false;
            F35.ObservedValue = false;
            M35.ObservedValue = false;
            MYSA.ObservedValue = false;
            FYSA.ObservedValue = false;
            MA.ObservedValue = false;
            FA.ClearObservedValue();
            GFA1.ObservedValue = false;
            GMA1.ObservedValue = true;
            Console.WriteLine();
            Console.WriteLine("  P(OSA | ¬S, ¬F35, ¬M35, ¬MYSA, ¬FYSA, ¬MA, ¬GFA1, GMA1) = " + ie.Infer(OSA));

            //GMA1 = true, OSA (female)
            S.ObservedValue = true;
            F35.ObservedValue = false;
            M35.ObservedValue = false;
            MYSA.ObservedValue = false;
            FYSA.ObservedValue = false;
            MA.ObservedValue = false;
            FA.ClearObservedValue();
            GFA1.ObservedValue = false;
            GMA1.ObservedValue = true;
            Console.WriteLine();
            Console.WriteLine("  P(OSA | S, ¬F35, ¬M35, ¬MYSA, ¬FYSA, ¬MA, ¬GFA1, GMA1) = " + ie.Infer(OSA));

            Console.ReadKey();
        }
    
    }
}
