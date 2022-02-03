from flask import render_template, Flask, request, redirect

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,GRU,Dense,LSTM
from tensorflow.keras.losses import sparse_categorical_crossentropy

import random
import numpy as np


app = Flask(__name__)

#
#
#   Home Page Code Below
#
#
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


#
#
# Bad PAssword Gen Code Below
#
#

vocab = [' ', '*', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', 'Z', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
char_to_ind = {char:ind for ind,char in enumerate(vocab)}
ind_to_char= np.array(vocab)

def sparse_cat_loss(y_true,y_pred):
    return sparse_categorical_crossentropy(y_true,y_pred,from_logits=True)

def create_model(vocab_size,embed_dim,rnn_neurons,batch_size):
    model = Sequential()
    
    model.add(Embedding(vocab_size,embed_dim,batch_input_shape=[batch_size,None]))
    model.add(GRU(rnn_neurons,return_sequences=True, stateful=True,recurrent_initializer='glorot_uniform'))
    model.add(Dense(vocab_size))
    
    model.compile(optimizer='adam',loss=sparse_cat_loss)
    
    return model

def generate_text(model,start_seed,gen_size=7,temp=1.0):
    num_generate = gen_size
    start_seed = ', ' + start_seed
    input_eval = [char_to_ind[s] for s in start_seed]
    
    input_eval = tf.expand_dims(input_eval,0)
    
    text_generated = []
    
    temperature = temp
    
    model.reset_states()
    
    for i in range(num_generate):
        preds = model(input_eval)
        preds = tf.squeeze(preds,0)
        preds = preds / temperature
        pred_id = tf.random.categorical(preds,num_samples=1)[-1,0].numpy()
        input_eval = tf.expand_dims([pred_id],0)
        text_generated.append(ind_to_char[pred_id])
        
    s = start_seed + "".join(text_generated)
    r = s.split(',')
    return r[1]

tstmodel = create_model(len(vocab),embed_dim=64,rnn_neurons=1026,batch_size=1)
tstmodel.load_weights('./PasswordGen.h5')
tstmodel.build(tf.TensorShape([1,None]))

@app.route('/bad_password_generator', methods=['GET', 'POST'])
def badpwGen():
    try:
        if request.method == 'GET':
            pw=None
        elif request.method=='POST':
            pw = generate_text(tstmodel, "", gen_size=13, temp=0.8)
    except:
        pw=None
    finally:
        return render_template('main.html', pw=pw)


#
#
#   Number Survival Code Below!
#
#
@app.route('/number_survival', methods=['GET'])
def number_survival():
    return render_template('number_home.html')

def get_choice():
    #Choose from add,subtract,multiply,divide
    op = random.choice(['a','s','m','d'])
    val = 0
    if op == 'a':
        val = random.randint(0, 30)
    elif op=='s':
        val=random.randint(-100,-1)
    elif op=='m':
        val = random.randint(0, 2)
    elif op=='d':
        #Remeber to not divide by zero!
        val =random.randint(1, 5)
    return op, val

def calc_health(user_val, op, val):
    print(f'{op}, {val:3}, {user_val}')
    if op=='a' or op=='s':
        user_val+=val
    elif op=='m':
        user_val*=val
    elif op=='d':
        user_val = user_val/val
    else:
        user_val=0
    return int(user_val)

@app.route('/game', methods=['GET','POST'])
def game():
    count = int(request.form['count']) + 1
    if count > 1:
        user_val = calc_health(int(request.form['user_val']), request.form['op'], int(request.form['val']))
    else:
        user_val=int(request.form['user_val'])

    if user_val<=0:
        return redirect(('/end/'+str(count)))

    op1, val1 = get_choice()
    op2, val2 = get_choice()

    if op1 == op2 and op1=='m' and val1 == val2 and val1==0:
        op2, val2 = get_choice()

    game_val=0 #calc_health(int(request.form['game_val']), op1, val1)

    # if game_val<=0:
    #     game_val=user_val
    
    # if count % 8 == 0:
    #     op1 = op2 = 's'
    #     val1 = val2 = random.randint(-1*(user_val-1),-1)

    return render_template('game.html', user_val=user_val,
                                        game_val=game_val,
                                        op1=op1,
                                        val1=val1,
                                        op2=op2,
                                        val2=val2,
                                        count=count)

@app.route('/end/<count>', methods=['GET','POST'])
def end(count):
    return render_template('end.html', count=count)

if __name__=='__main__':
    app.run()