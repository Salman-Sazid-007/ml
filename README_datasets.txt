=========================================================================
 6 REAL DATASETS FOR PRACTICE  (seaborn-data, GitHub)
=========================================================================

FILE            SHAPE        RECOMMENDED TARGET   TASK             NOTES
-----------------------------------------------------------------------
iris.csv        150 x 5      species              classification   clean, 3 class, best first test
penguins.csv    344 x 7      species              classification   19 nulls, 3 text cols, 3 class
titanic.csv     891 x 15     survived             classification   869 nulls!, 107 dup, LEAK ache
tips.csv        244 x 7      tip                  REGRESSION       4 text cols
mpg.csv         398 x 9      mpg                  REGRESSION       6 nulls, 'name' = ID column
diamonds.csv    53940 x 10   price                REGRESSION       boro data, SAMPLE use koro

-------------------------------------------------------------------------
IMPORTANT: target SHESH column e NEI (titanic, tips, mpg, diamonds e)
   - titanic: survived = 1st column,  shesh column 'alone'
   - tips   : tip      = 2nd column,  shesh column 'size'
   - mpg    : mpg      = 1st column,  shesh column 'name'
   - diamonds: price   = 7th column,  shesh column 'z'
   => shudhu shesh column dhore nile VUL hobe. columns print kore dekho.

-------------------------------------------------------------------------
TITANIC - DATA LEAKAGE WARNING (khub important):
   'alive' column ta 'survived' er e onno rup (yes/no vs 1/0)
   'class' column ta 'pclass' er e onno rup
   eguli na sorale model 100% accuracy dibe - eta VUL, cheating.

   thik kora:
       df = df.drop(columns=["alive", "class", "who", "adult_male"])

   proman:  leak soho    = 1.000 accuracy  (mithya)
            leak chara    = 0.764 accuracy  (asol)

-------------------------------------------------------------------------
MPG - 'name' column ta car er nam, 305 ta alada -> ID er moto, bad dao:
       df = df.drop(columns=["name"])

DIAMONDS - 53940 row, SVM khub slow hobe. choto koro:
       df = df.sample(5000, random_state=42)

-------------------------------------------------------------------------
UNSUPERVISED practice er jonno: target column ta bad diye chalao
       df = df.drop(columns="species")     # penguins
   penguins clustering e valo kaj kore (3 ta species alada hoy)
=========================================================================
