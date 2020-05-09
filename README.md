# Topic Modeling for Iran Stock Market data of Telegram Posts
Topic modeling of datasets of telegram posts about Tehran Stock Market's symbols and industries using LDA and lda2vec.

Link of Codes on [google colab](https://drive.google.com/file/d/17aCmKkY147Bdn01_KBJ8XlrS0KOULAFC/view?usp=sharing).

----

There are 4 more popular methods for topic modeling:
  - **LSA**: Latent Semantic Analysis
  -- Building Document-Term Matrix
-- Convert to TF-IDF scores
-- Decomposing Document-Term Matrix to Document-Topic and Term-Topic Matrices Using Truncated SVD
  - **pLSA** : Probabilistic Latent Semantic Analysis
  -- Like LSA otherwise, Probabilistic Technique to Matrix Decomposition Instead of SVD
  - **LDA** : 
  -- Bayesian Version of pLSA
  - ****LDA2VEC****
  -- Deep Learning Version of LDA
 -- Extension of LDA and Word2Vec
-- This Method First Presented in [Chris Moody's paper](https://arxiv.org/abs/1605.02019)

# Implementations
There is some implementation of LDA2VEC. I reused the implementation of [LDA2VEC by PyTorch](https://github.com/TropComplique/lda2vec-pytorch). 

As soon as, we want to just represent the term-topic of each document. we just need **Gensim** for loading *models.Lda2Model* as **LDA** part and  *models.Word2Vec* as **VEC** part!

There are implementations of both LDA and lda2vec.
# Parameters
- MIN_COUNTS 
 - MAX_COUNTS 
 words with count < MIN_COUNTS and count > MAX_COUNTS  will be removed
- n_topics 
Number of the topic we want to extract from a bunch of docs
 - MIN_LENGTH 
 minimum document length, meaning, number of words, after preprocessing

- HALF_WINDOW_SIZE 
 half the size of the context around a word. it must be that 2*HALF_WINDOW_SIZE < MIN_LENGTH

#  Attempts
- I had trained word-embedding on whole posts But created LDA for each symbol or industry. so as to get richer word-embedding, but the result was disappointing! I think This is no the wrong way and the fault was somewhere else!
- I changed the strategy to separate data according to industries and symbols. Then train word2vec on each industry or symbol, then create an LDA model on that.
 - after some different config testing, I come to MAX_COUNTS. It really gets the result better.
 
# Execution
- All time-intensive operations have dumped to NumPy and pickles.
# Result
## Industry = IREXBANKSCREDITINSTITUTIONS
### LDA
topic 0 : تسهیلات محل توجه سهامدار جهت فرابورس دلار تسویه صفوف کشور

topic 1 : صورت سازمان فرابورس میلیون پول اقتصاد توجه موضوع صفوف سهامدار

topic 2 : فرابورس خبر سهامدار قرار صفوف میلیون قیمت موضوع کاهش کار

topic 3 : صفوف اقتصاد فرابورس پول میلیون دلار تومان صادرات انجام محل

topic 4 : فرابورس صفوف میلیون خبر موضوع تومان پول سازمان جهت کار

topic 5 : فرابورس مثبت صفوف قیمت پول موضوع کشور دلار صورت انجام

topic 6 : صفوف فرابورس توجه کشور فولاد نفت صورت تومان اقتصاد موضوع

topic 7 : فرابورس صفوف میلیون اقتصاد فولاد تومان دولت سهامدار قرار ملت

topic 8 : میلیون اقتصاد تاریخ گروه کشور فرابورس پتروشیمی صفوف خبر جهت

topic 9 : توجه تهران خبر فرابورس قرار میلیون اقتصاد عرضه کشور پول

### lda2vec
topic 1 : تسهیلات تسویه ساختمان توجه جهت متعلق بابت نوین تاریخ مذکور

topic 2 : کشور صادرات انجام معدن اعلام میلیون صورت درصد مرکزی ملت

topic 3 : مثبت منفی پول پرداختی موضوع خبر کرد#کن انها تراز میکنیم

topic 4 : فرابورس کرمان مورخ ملت پرداختی توجه تاریخ تراز میلیون دستگاه

topic 5 : میلیون ملت صادرات درآمد مورخ تومان کشور زیان انجام سهامدار

topic 6 : صادرات نشان دلار کشور سهامدار شرایط داخل تاریخ موضوع میدهد

topic 7 : صادرات دستگاه آبان میلیون سفار فروخته آذر مقدار شمش صادر

topic 8 : فرابورس ملت وبملت سفار شیر غاذر پترول ونوین وتجارت همراه

topic 9 : میلیون صادرات تاریخ محل پارس طرح نقد گروه صنایع جهت

topic 10 : نفت تهران قیمت درصد کشور میلیون توجه دلار عرضه عموم



## Symbol= "IRO1BVMA0001"
## ('ما', 'بیمه ما')
### LDA

topic 0 : نکته بازی میکنیم میدهد اتفاق نشان کشور داشت#دار قرار عرضه

topic 1 : داشت#دار منفی انها میکنیم قرار اتفاق فضا عزیز بازی نشان

topic 2 : خرید قرار نکته نفت بانک قیمت اقتصاد شرکت عزیز بنیاد

topic 3 : انها بحث اتفاق میکند نکته قیمت میلیارد ادامه سیاسی ریال

topic 4 : میلیارد قرار شرکت حرف میکنیم عزیز فضا نشان کشور درست

topic 5 : عزیز داشت#دار قرار افزایش عرضه اقتصاد میکنیم اتفاق انها فضا

topic 6 : افزایش داد#ده اقتصاد نکته شرکت عزیز بازی میکند بحث عرضه

topic 7 : شرکت درست بازی اتفاقات اتفاق شد#شو انها قیمت بنیاد فضا

topic 8 : عزیز حرف میکنیم اتفاق کشور اقتصاد نشان میکند قرار شرکت

topic 9 : کشور انها عزیز داشت#دار قرار شرکت بازی نکته زد#زن داد#ده


### ida2vec
topic 1 : اقتصاد تولید کشور شرکت انجام افزایش میلیارد پتروشیمی مردم کرونا

topic 2 : منفی انها بازی ریزش فضا بنیاد افت مسیر نشان میکند

topic 3 : بانک نفت ارز تومان میلیارد اقتصاد شرکت درصد مردم قرار

topic 4 : نکته ماهه قیمت نشان ریال اصلاح شاخص میدهد وضعیت افت

topic 5 : میلیارد شرکت ریال دستگاه تولید تومان نفت افزایش پایان کشور

topic 6 : داشت#دار عزیز مردم اقتصاد داد#ده میکنیم اولیه قرار گفت#گو امریکا

topic 7 : شرکت افزایش تولید پایان سهام اقتصاد درصد بانک انجام صورت

topic 8 : اتفاقات اتفاق بازی شد#شو میکنیم سیاسی هفته منفی نکته بحث

topic 9 : اقتصاد داد#ده عزیز #هست ادامه گرفت#گیر برجا سازمان پیشنهاد حرف

topic 10 : آمریکا کشور برجا تعهدات توافق اروپا منطقه جمهوری انجام اقدامات


# Failed Attempt
At first, I misunderstood the problem as sentiment analysis, As it is common to analyzing sentiment of stock market activists' reviews of symbols to predict sales and buy queues. As dataset was unlabeled and large. I studied many things about how to accomplish this task:

- Unsupervised Sentiment Analysis
    * [Sentiment Analysis Clustering](https://towardsdatascience.com/unsupervised-sentiment-analysis-a38bf1906483)
    -- The idea is to apply word2vec word embedding on data. then to cluster data into two clusters, meaning binary clustering using K-means. Then define sentiment score using a distance measure. after a specificity score for a word in the document using TF-IDF scores,  developing a model to score each document using words sentiments score and specificity.
    * [Unsupervised Sentiment Analysis Tools](https://medium.com/@Intellica.AI/vader-ibm-watson-or-textblob-which-is-better-for-unsupervised-sentiment-analysis-db4143a39445):
    -- There are some pre-trained S.A. tools for English language like **TextBlob** and **Vader** and **IBM Watson**. Watson is a commercial solution, but the two first are for free. All of these techniques are applicable to English!

- Labeling Large Dataset:
    * Random Labeling
    -- Forget the tools name! something like Anura!
    * Active learning:
    -- Interactive labeling of data using ML.
    * Text annotation tools:
    -- I tested **[doccano](https://github.com/doccano/doccano)** on an ubuntu server in an intelligent way to train the model on small labeled data then test unlabeled data with model and correct wrong label!

**Caution:** Most of telegram posts don't imply any positive or negative sentiment, thus those are neutral.
