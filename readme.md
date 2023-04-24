## Korean to English Translation Project

### 1. Project Description
- This project is for translating Korean to English.
- The dataset comes from kaggle (https://www.kaggle.com/datasets/rareloto/naver-dictionary-conversation-of-the-day?resource=download)
- `kaggle datasets download -d rareloto/naver-dictionary-conversation-of-the-day`

>The project run result: 
- <img src="/image/img_1.png" width="400" height="300">


### 2. Data Description
- the conversations.csv saved in is as follows：
- <img src="/image/img.png" width="500" height="300">

We only get the column where the Korean language is and the column where the English language is in the file
，and save it as a selected_conversations.txt file
- The dataset has a total of 4563 sentences


### 3. File Description

- **main/main.py:** the main file of the project

- **main/lib.py:**  the file where the function is stored
- **main/readData.py:** create .txt data file from .csv file
- **data:** the folder where the dataset is stored
- **image:** the folder where the image is stored
- **requirements.txt:** save running environment information


### 4. How to run the code
- environment settings
- In principle, only installed torch and numpy can run the code
- if can't run the code, you can:
```pip install -r requirements.txt```
- run the code ```python main.py```
- if you want to train by yourself , you can set the parameter ```train = True``` in the main.py file

### 5. Code Description

> The Transformer model is created using the self-attention mechanism.
Transformer is composed of encoder and decoder.
>
    
* Positionwise feed-forwad network: 

  *PositionWiseFFN()*
  * The shape of the input X (batch size, number of time steps or sequence length, number of hidden units or feature dimension)
  * will be converted by a two-layer perceptron into an output tensor of shape (batch size, number of time steps, ffn_num_outputs)



* Add and Norm:
    
    * nn.LayerNorm()
    * nn.BatchNormld()

* Encoder:
  * EncoderBlock()：The EncoderBlock class contains two sublayers: multi-head self-attention and position-based feedforward networks, both of which use residual connections followed by layer normalization.
  * TransformerEncoder()：Stacked num_layers instances of the EncoderBlock class

* Decoder:
  * DecoderBlock()： Each layer implemented consists of three sublayers: decoder self-attention, "encoder-decoder" attention, and position-based feedforward network
  * TransformerDecoder()：A complete Transformer decoder consisting of num_layers DecoderBlock instances


* hyperparameters：
  * num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10 
  * lr, num_epochs, device = 0.0002, 4000, d2l.try_gpu()
  * ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
  * key_size, query_size, value_size = 32, 32, 32  
  * norm_shape = [32]