## 说明
* 1、captcha_train.py是原来的网络结构，调用的相应captcha_model.py。
   四层（卷积+池化）+两层全连接。
   使用captcha_eval.py评估的准确率为78%，训练集大小为30000，测试集5000，,80000轮训练

* 2、captcha_traina.py是原来的网络结构，调用的相应captcha_modela.py。
   （卷积+池化）+两层全连接.
   使用captcha_eval.py评估的准确率为89%，训练集大小为30000，测试集5000，,100000轮训练

* 3、captcha_model (normalization).py是在 2 的基础上在卷积层后，全连接层前加一个归一化。
   好处是在 8000 步内收敛，收敛较快。
