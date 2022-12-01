# تولید شعر با شبکه عصبی LSTM و تنسورفلو

![](https://files.virgool.io/upload/users/1223901/posts/rynq4emx1qcx/5bqtxkgjyhop.jpeg)

<p>یادگیری عمیق ابزاری فوق العاده برای ساخت پروژه های مهیج و سرگرم‌کننده است، هر روز خبری در مورد کاربرد های جدید یادگیری عمیق منتشر میشود و همگان را در تعجب فرو میبرد، آموزش این مدل های جذاب هم در نوع خودش سرگرم‌کننده است مخصوصا مدل های مرتبط به پردازش زبان طبیعی(NLP).

در این ویرگول میخواهم نشان بدهم چگونه میتوان از یادگیری عمیق برای تولید شعر استفاده کنیم، مدلی که خواهیم ساخت، ارتباط بین کارکتر ها را متوجه میشود و احتمال وقوع کارکتر بعدی را محاسبه میکند.</p>

>در این پروژه از شبکه عصبی LSTM استفاده خواهیم کرد. به طور خلاصه، LSTM یک نوع خاص از یک شبکه عصبی RNN است که در پردازش داده هایی که دارای توالی منظم و مرتبط هستند استفاده میشود. مثل متون، موسیقی ها، ویدئو ها و...</br>
تفاوت RNN ها با شبکه های عصبی کلاسیک داشتن وابستگی زمانی یا اثر حافظه است که داده های قبلی را در حافظه به خاطر میسپارد تا در تصمیم گیری های بعدی استفاده کند.</br>
[برای مطالعه بیشتر](https://l.vrgl.ir/r?l=https%3A%2F%2Fcolah.github.io%2Fposts%2F2015-08-Understanding-LSTMs%2F&st=post&si=rynq4emx1qcx&k=qToANxWsJZcbp30NlTE%2FGPXPRobbWIVbY2orsTBmE0E%3D).

## آماده سازی
</br>

<p>برای شروع به تنسورفلو و نامپای نیاز خواهیم داشت که میتوانید با دستور زیر آنها را نصب کنید :</p>

> pip install -U tensorflow-gpu</br>
> pip install -U numpy

<p>ما از دیوان حافظ استفاده میکنیم اما میتوانید برای دقت بیشتر از اشعار سعدی یا فردوسی استفاده کنید(یا هرنوع متن بالای 100 هزار کارکتر)، اشعار حافظ را با دستور زیر دریافت میکنیم :</p>

>wget -O hafez.txt https://raw.githubusercontent.com/amnghd/Persian_poems_corpus/master/normalized/hafez_norm.txt

<p>ابزار های مورد نیاز را ایمپورت میکنیم و سپس محتوای کتاب را میخوانیم :</p>

>import tensorflow as tf</br>
import keras</br>
from keras.layers import  Input, LSTM, Dense</br>
import tensorflow.keras.optimizers as optimizers</br>
import numpy as np</br>
import random<br>
</br>
text = open(&quothafez.txt&quot, &quotr&quot, encoding=&quotutf-8&quot).read()

## پردازش متن و ایجاد دیتاست
</br>

<p>شبکه عصبی نمیتواند داده ها به صورت متنی دریافت کند، برای همین باید کارکتر ها را به اعداد صحیح تبدیل کنیم :</p>

>chars = sorted(list(set(text)))</br>
char_indices = dict((c, i) for i, c in enumerate(chars))</br>
indices_char = dict((i, c) for i, c in enumerate(chars))

<p>برای تولید متن مدل با دیدن کارکتر های قبل یاد میگیرد که کارکتر بعدی کدام است :</p>

![](https://files.virgool.io/upload/users/1223901/posts/rynq4emx1qcx/5378m1t6nquv.jpeg)

<p>پس با توجه به تصویر بالا، input ها یک ایندکس از آخر عقب تر از target ها هستند و target ها نیز یک ایندکس از اول جلوتر از input ها هستند، پس به این شیوه متغییر های x و y رو میسازیم:</p>

>maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step): </br>
    sentences.append(text[i : i + maxlen])</br>
    next_chars.append(text[i + maxlen])</br>
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)</br>
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)</br>
for i, sentence in enumerate(sentences):</br>
    for t, char in enumerate(sentence):</br>
        x[i, t, char_indices[char]] = 1</br>
    y[i, char_indices[next_chars[i]]] = 1

## مدل سازی
</br>

<p>حالا میتوانیم مدل LSTM خود را تعریف کنیم. در اینجا از یک لایه LSTM با 128 واحد حافظه به عنوان حافظه پنهان و لایه خروجی یک Dense layer هست که از تابع softmax استفاده میکند، برای بهینه ساز هم از Adam استفاده میکنیم :</p>

>model = keras.Sequential(</br>
    [</br>
        keras.Input(shape=(maxlen, len(chars))),</br>
        layers.LSTM(128),</br>
        layers.Dense(len(chars), activation=&quotsoftmax&quot),</br>
    ]</br>
)</br>
optimizer = optimizers.Adam(learning_rate=0.01)</br>
model.compile(loss=&quotcategorical_crossentropy&quot, optimizer=optimizer)

<p>به دلیل اینکه مدل برای پیش‌بینی کارکتر بعدی، احتمال هر کارکتر را برمیگرداند، یک تابع برای text sampling نیاز خواهیم داشت :</p>

>def sample(preds, temperature=1.0):</br>
    # helper function to sample an index from a probability array</br>
    preds = np.asarray(preds).astype(&quotfloat64&quot)</br>
    preds = np.log(preds) / temperature</br>
    exp_preds = np.exp(preds)</br>
    preds = exp_preds / np.sum(exp_preds)</br>
    probas = np.random.multinomial(1, preds, 1)</br>
    return np.argmax(probas)

<p>و در آخر آموزش مدل را آموزش میدهیم :</p>

>epochs = 40</br>
batch_size = 128</br>
</br>
model.fit(x, y, batch_size=batch_size, epochs=40)

<p>آموزش مدل روی NVIDIA Tesla K50 حدود دو دقیقه طول کشید. حالا میتوانیم مدل را تست کنیم :</p>

>for i in range(10):</br>
    start_index = random.randint(0, len(text) - maxlen - 1)</br>
    for diversity in [0.2, 0.5, 1.0, 1.2]:</br>
        print(&quot...Diversity:&quot, diversity)</br>
</br>
        generated = &quot&quot</br>
        sentence = text[start_index : start_index + maxlen]</br>
        print('...Generating with seed: &quot' + sentence + '&quot')</br>
</br>
        for i in range(400):</br>
            x_pred = np.zeros((1, maxlen, len(chars)))</br>
            for t, char in enumerate(sentence):</br>
                x_pred[0, t, char_indices[char]] = 1.0</br>
            preds = model.predict(x_pred, verbose=0)[0]</br>
            next_index = sample(preds, diversity)</br>
            next_char = indices_char[next_index]</br>
            sentence = sentence[1:] + next_char</br>
            generated += next_char</br>
</br>
        print(&quot...Generated: &quot, generated)</br>
        print()

<p>با اجرای کد، ده متن با diversity های مختلف تولید میشود :</p>

![](https://files.virgool.io/upload/users/1223901/posts/rynq4emx1qcx/plrkjthnkq7a.png)

<p>متن ورودی به صورت تصادفی از خود متن اصلی انتخاب میشود اما شما میتوانید متن خود را به مدل بدهید.</p>

## بررسی اشعار تولید شده
</br>

>به خاک نشینی که بر سر خود از در آن چه اند</br>
گر چه ما به دست است و دل بر این خواهد شد

>ما نگه از دوست ما به در میان بود</br>
به رخسان من و ماه می و مهر ما را

>هر که به یاد به باد صبا به میان بود</br>
به خون می کن که زیرکش که من می برد از او

>که در دل از او تو به ملامت می برد</br>
تو به پیش کام دل ما نازک سخن بده

>از لب گل صفیه و دارد سحر پرده کنم</br>
حافظ ار خاطر و خال ما و ز بر بخت نیک

## بهبود مدل
</br>

<p>برای بهبود نتایج مدل چند کار میشه انجام داد :</p>

1. استفاده از مدل پیچیده تر(در اینجا ما فقط از یک لایه LSTM آنهم با 128 واحد حافظه استفاده کرده ایم، استفاده از LSTM های بیشتر و لایه Dropout نتیجه را بهبود میبخشد).
2. متن بهتر(متنی که استفاده کردیم حدود 300 هزار کارکتر داشت، توصیه میشه برای داشتن مدل بهتر از متنی با بیش از یک میلیون کارکتر استفاده شود).

<p>نوت بوک این مطلب روی گیت هاب به آدرس زیر موجود است:</p>

## منابع
</br>

1. [https://keras.io/examples/generative/lstm_character_level_text_generation](https://l.vrgl.ir/r?l=https%3A%2F%2Fkeras.io%2Fexamples%2Fgenerative%2Flstm_character_level_text_generation&st=post&si=rynq4emx1qcx&k=zOeyx437OR3QeCv%2FjB2XsCf%2FKnNqtlaItVc2Q19BFN8%3D).
2. [https://www.tensorflow.org/text/tutorials/text_generation](https://l.vrgl.ir/r?l=https%3A%2F%2Fwww.tensorflow.org%2Ftext%2Ftutorials%2Ftext_generation&st=post&si=rynq4emx1qcx&k=Xo01%2F3zx8TbOD5M81Hr6UKoAcHyw2%2BJdd4Nn9ESMbxg%3D)
3. [https://karpathy.github.io/2015/05/21/rnn-effectiveness](https://l.vrgl.ir/r?l=https%3A%2F%2Fkarpathy.github.io%2F2015%2F05%2F21%2Frnn-effectiveness%2F&st=post&si=rynq4emx1qcx&k=EziJtl8Ly0STV%2F5NYRTMCZtbkrG7r%2FW42DFNlIUC0nA%3D)
4. Natural Language Processing IN Action, Manning Publication
5. [https://github.com/amnghd/Persian_poems_corpus](https://l.vrgl.ir/r?l=https%3A%2F%2Fgithub.com%2Famnghd%2FPersian_poems_corpus&st=post&si=rynq4emx1qcx&k=vqnt5%2F3s7HTiF0xPTRd3SBaWSAgB%2BmuEJlpNoYX5QIU%3D)
