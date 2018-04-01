import numpy as np
import librosa
import os
from collections import Counter
import tensorflow as tf
import time

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class SignalProcessing(object):
    def __init__(self, sample_rate, frame_length, frame_step, cut_ratio, window_type, log_offset,
                 lower_edge_hertz=80.0, upper_edge_hertz=7600, num_mel_bins=64, num_mfccs=20):
        # assuming that fft_length == frame_length
        # if frame_length : frame_step = 3:1, then the window could make the signal reconstructed would have the same magnitude; otherwise it would be greater
        self.signals = tf.placeholder(shape=[None, None], dtype=tf.float32)
        self.frames = tf.contrib.signal.frame(self.signals, frame_length=frame_length, frame_step=frame_step)
        if window_type is None:
            self.reconstructed_signals = tf.contrib.signal.overlap_and_add(self.frames, frame_step=frame_step)
        elif window_type == 'hamming':
            mid_frames = self.frames * tf.contrib.signal.hamming_window(frame_length)
            self.reconstructed_signals = tf.contrib.signal.overlap_and_add(mid_frames, frame_step=frame_step)
        elif window_type == 'hanning':
            mid_frames = self.frames * tf.contrib.signal.hanning_window(frame_length)
            self.reconstructed_signals = tf.contrib.signal.overlap_and_add(mid_frames, frame_step=frame_step)

        self.stfts = tf.contrib.signal.stft(self.signals, frame_length=frame_length,
                                            frame_step=frame_step, fft_length=frame_length)
        self.magnitude_spectrograms = tf.abs(self.stfts)
        self.power_spectrograms = tf.real(self.stfts * tf.conj(self.stfts))
        self.spectrogram_patches = tf.contrib.signal.frame(
            self.magnitude_spectrograms, frame_length=frame_length // cut_ratio, frame_step=frame_step // cut_ratio,
            axis=1
        )
        self.log_magnitude_spectrogram = tf.log(self.magnitude_spectrograms + log_offset)

        num_spectrogram_bins = self.magnitude_spectrograms.shape[-1].value
        linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
            num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz
        )
        self.mel_spectrograms = tf.tensordot(self.magnitude_spectrograms, linear_to_mel_weight_matrix, 1)
        self.mel_spectrograms.set_shape(
            self.magnitude_spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

        self.log_mel_spectrograms = tf.log(self.mel_spectrograms + log_offset)
        self.mfccs = tf.contrib.signal.mfccs_from_log_mel_spectrograms(
            self.log_mel_spectrograms
        )[..., :num_mfccs]


# 训练样本路径
wav_path = 'wav/train'
label_file = 'doc/trans/train.word.txt'

# 获得训练用的wav文件路径列表
def get_wav_files(wav_path=wav_path):
    wav_files = []
    for (dirpath, dirnames, filenames) in os.walk(wav_path):
        for filename in filenames:
            if filename.endswith('.wav') or filename.endswith('.WAV'):
                filename_path = os.sep.join([dirpath, filename])
                if os.stat(filename_path).st_size < 240000:  # 剔除掉一些小文件
                    continue
                wav_files.append(filename_path)
    return wav_files


wav_files = get_wav_files()[:100]

# 读取wav文件对应的label
def get_wav_lable(wav_files=wav_files, label_file=label_file):
    labels_dict = {}
    with open(label_file, 'r') as f:
        for label in f:
            label = label.strip('\n')
            label_id = label.split(' ', 1)[0]
            label_text = label.split(' ', 1)[1]
            labels_dict[label_id] = label_text
    labels = []
    new_wav_files = []
    for wav_file in wav_files:
        wav_id = os.path.basename(wav_file).split('.')[0]
        if wav_id in labels_dict:
            labels.append(labels_dict[wav_id])
            new_wav_files.append(wav_file)
    return new_wav_files, labels


wav_files, labels = get_wav_lable()
print("样本数:", len(wav_files))  # 8911
print(wav_files[0], labels[0])
# wav/train/A11/A11_0.WAV -> 绿 是 阳春 烟 景 大块 文章 的 底色 四月 的 林 峦 更是 绿 得 鲜活 秀媚 诗意 盎然
test_wav_files = get_wav_files(wav_path='wav/test')
test_wav_files, test_labels = get_wav_lable(wav_files=test_wav_files, label_file='doc/trans/test.word.txt')
print(len(test_wav_files), len(test_labels))
for i in range(10):
    print(test_wav_files[i], test_labels[i])

# 词汇表(参看练习1和7)
all_words = []
for label in labels:
    all_words += [word for word in label]
counter = Counter(all_words)
count_pairs = sorted(counter.items(), key=lambda x: -x[1])

count_pairs = count_pairs[ 1 : ]

#print(count_pairs)

words, _ = zip(*count_pairs)
words_size = len(words)
print('词汇表大小:', words_size)

word_num_map = dict(zip(words, range(len(words))))

assert ' ' not in word_num_map

labels_vector = []
for lb in labels:
    tmplb = []
    for tl in lb:
        if tl != ' ':
            tmpx = word_num_map[tl]
            tmplb.append(tmpx)
    labels_vector.append(tmplb)

num_word_map = {}
for w in word_num_map:
    n = word_num_map[w]
    num_word_map[n] = w


print("file", wav_files[0])
print(len(labels_vector[0]), "vector", labels_vector[0])
decode_string = ""
for i in labels_vector[0]:
    tmpw = num_word_map[i]
    decode_string += tmpw
print("decode result", decode_string)
print(len(labels[0]), "label", labels[0])

label_max_len = np.max([len(label) for label in labels_vector])
print('最长句子的字数:', label_max_len)

mfcc_max_len = 680
n_classes = len(word_num_map) + 2

# max len is about 350000
sp = SignalProcessing(16000, 2048, 512, 1, None, 1e-12)
# 2048 512 the max time_step is 680 and the frequency is 1025((2048+1/2))

# read wavs
wavs = []
count = 0
for w in wav_files:
    wav, sr = librosa.load(w, mono=True)
    nw = np.zeros(350000)
    nw[:len(wav)] = wav
    wavs.append(nw)
    count += 1
    if count % 50 == 0:
        print("wav {} has readed".format(count))

wl = len(wavs)
# conpute the whole spectrograms
batch_size = 50
rounds = wl // batch_size
remain = wl % batch_size
wav_spectrograms = None
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as session:
    for i in range(rounds):
        cur_w = wavs[i*batch_size : (i+1)*batch_size]
        spec = session.run(sp.log_magnitude_spectrogram, feed_dict={sp.signals : cur_w})
        if wav_spectrograms is None:
            wav_spectrograms = spec
        else:
            wav_spectrograms = np.concatenate((wav_spectrograms, spec), axis=0)
        print("{} specs has computed".format((i+1)*batch_size))
    if remain > 0:
        cur_w = wavs[-remain : ]
        spec = session.run(sp.log_magnitude_spectrogram, feed_dict={sp.signals: cur_w})
        if wav_spectrograms is None:
            wav_spectrograms = spec
        else:
            wav_spectrograms = np.concatenate((wav_spectrograms, spec), axis=0)

print("data of spectrogram", wav_spectrograms.shape)
tf.reset_default_graph()

batch_size = 20
if batch_size > len(wav_spectrograms):
    batch_size = len(wav_spectrograms)
n_batch = int(np.ceil(len(wav_spectrograms) / batch_size))

len_data = len(wav_spectrograms)

def decode_a_seq(indexes, spars_tensor):
    decoded = []
    for m in indexes:
        tmpi = spars_tensor[1][m]
        tmps = num_word_map[tmpi]
        decoded.append(tmps)
    return decoded

def decode_sparse_tensor(sparse_tensor):
    #print("sparse_tensor = ", sparse_tensor)
    decoded_indexes = list()
    current_i = 0
    current_seq = []
    for offset, i_and_index in enumerate(sparse_tensor[0]):
        i = i_and_index[0]
        if i != current_i:
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
        current_seq.append(offset)
    decoded_indexes.append(current_seq)
    #print("decoded_indexes = ", decoded_indexes)
    result = []
    for index in decoded_indexes:
        #print("index = ", index)
        result.append(decode_a_seq(index, sparse_tensor))
        #print(result)
    return result

def sparse_tuple_from(sequences, dtype=np.int32):
    """
    Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []
    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
    return indices, values, shape


def Get_Batch(batch_size):
    sltidx = np.random.choice(len_data, batch_size)
    data = wav_spectrograms[sltidx]
    lblvec = []
    lbltxt = []
    for idx in sltidx:
        lblvec.append(labels_vector[idx])
        lbltxt.append(labels[idx])
    sparse_targets = sparse_tuple_from(lblvec)
    seq_len = np.ones(batch_size) * mfcc_max_len
    # return feature data, sparse_data, seq_len and lbl_txt
    return data, sparse_targets, seq_len, lbltxt

# LSTM
num_hidden = 256

def leaky_relu(x, alpha=0.01):
    return tf.maximum(x, alpha*x)

def Conv2dLayer(number, input_data, filters, kernel_size, reuse=False):
    with tf.variable_scope("conv2d_layer_{}".format(number), reuse=reuse):
        conv = tf.layers.conv2d(input_data, filters=filters, kernel_size=kernel_size, padding='same')
        lrl = leaky_relu(conv)
        pool = tf.layers.max_pooling2d(lrl, 2, [1,2], padding='same')
        return pool

def Conv1dLayer(number, input_data, filters, kernel_size, reuse=False):
    with tf.variable_scope('conv1d_layer_{}'.format(number), reuse=reuse):
        conv = tf.layers.conv1d(input_data, filters=filters, kernel_size=kernel_size, padding='same')
        return conv

def Get_Train_Model(num_lstm_layers=1):
    input = tf.placeholder(tf.float32, [None, 680, 1025])
    targets = tf.sparse_placeholder(tf.int32)
    seq_len = tf.placeholder(tf.int32, [None])

    input_layer = tf.reshape(input, (-1, 680, 1025, 1))
    conv1 = Conv2dLayer(1, input_layer, 1, [9,11])
    print("conv1", conv1)
    # bs 680 513 1
    conv2 = Conv2dLayer(2, conv1, 8, 7)
    print("conv2", conv2)
    # bs 680 257 8
    conv3 = Conv2dLayer(3, conv2, 16, 5)
    print("conv3", conv3)
    # bs 680 129 16
    conv4 = Conv2dLayer(4, conv3, 32, 5)
    print("conv4", conv4)
    # bs 680 65 32
    conv5 = Conv2dLayer(5, conv4, 64, 5)
    print("conv5", conv5)
    # bs 680 33 64
    conv6 = Conv2dLayer(6, conv5, 128, 3)
    print("conv6", conv6)
    # bs 680 17 128
    conv7 = Conv2dLayer(7, conv6, 128, 3)
    print("conv7", conv7)
    # bs 680 9 128
    conv8 = Conv2dLayer(8, conv7, 192, 3)
    print("conv8", conv8)
    # bs 680 5 192
    conv9 = Conv2dLayer(9, conv8, 256, 3)
    print("conv9", conv9)
    # bs 680 3 256
    conv10 = Conv2dLayer(10, conv9, 320, 3)
    print("conv10", conv10)
    # bs 680 2 320
    conv11 = Conv2dLayer(11, conv10, 320, 3)
    print("conv11", conv11)
    # bs 680 1 320
    outputs = tf.squeeze(conv11, 2)
    # bs 680 320

    #cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
    #outputs, _ = tf.nn.dynamic_rnn(cell, input, seq_len, dtype=tf.float32)

    shape = tf.shape(input)
    batch_s, max_timesteps = shape[0], shape[1]

    outputs = tf.reshape(outputs, [-1, 320])

    logits = tf.layers.dense(outputs, n_classes)
    logits = tf.reshape(logits, [batch_s, -1, n_classes])

    logits = tf.transpose(logits, (1, 0, 2))

    return logits, input, targets, seq_len

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(0.001,
                                                global_step,
                                                500,
                                                0.9,
                                                staircase=True)
logits, inputs, targets, seq_len = Get_Train_Model()
loss = tf.nn.ctc_loss(labels=targets,inputs=logits, sequence_length=seq_len)
cost = tf.reduce_mean(loss)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,global_step=global_step)
decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)

acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

init = tf.global_variables_initializer()


def report_accuracy(decoded_list, test_targets):
    print("")
    original_list = decode_sparse_tensor(test_targets)
    detected_list = decode_sparse_tensor(decoded_list)
    true_numer = 0

    if len(original_list) != len(detected_list):
        print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
              " test and detect length desn't match")
        return
    print("T/F: original(length) <-------> detectcted(length)")
    for idx, number in enumerate(original_list):
        detect_number = detected_list[idx]
        hit = (number == detect_number)
        print(hit, number, "(", len(number), ") <-------> ", detect_number, "(", len(detect_number), ")")
        if hit:
            true_numer = true_numer + 1
    print("Test Accuracy:", true_numer * 1.0 / len(original_list))

def do_report(session):
    print("*** do report ***")
    test_inputs, test_targets, test_seq_len, txts = Get_Batch(batch_size//8)
    test_feed = {inputs: test_inputs,
                 targets: test_targets,
                 seq_len: test_seq_len}
    dd, log_probs = session.run([decoded[0], log_prob], test_feed)
    report_accuracy(dd, test_targets)

def do_batch(session):
    train_inputs, train_targets, train_seq_len, txts = Get_Batch(batch_size)
    feed = {inputs: train_inputs, targets: train_targets, seq_len: train_seq_len}
    b_loss, b_targets, b_logits, b_seq_len, b_cost, steps, _ = session.run(
            [loss, targets, logits, seq_len, cost, global_step, optimizer], feed)

    print(b_cost, steps)
    #if steps > 0 and steps % 25 == 0:
        #do_report(session)

    return b_cost, steps


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as session:
    session.run(init)
    for curr_epoch in range(10000):
        print("Epoch.......", curr_epoch)
        train_cost = train_ler = 0
        for batch in range(100):
            start = time.time()
            c, steps = do_batch(session)
            train_cost += c * batch_size
            seconds = time.time() - start
            print("Step:", steps, ", batch seconds:", seconds)

        train_cost /= (100*batch_size)
        print(" to next")
        train_inputs, train_targets, train_seq_len, txts = Get_Batch(batch_size//8)
        val_feed = {inputs: train_inputs,
                    targets: train_targets,
                    seq_len: train_seq_len}
        print("to compute")
        if train_cost < 10:
            val_cost, val_ler, lr, steps = session.run([cost, acc, learning_rate, global_step], feed_dict=val_feed)
        else:
            val_cost, lr, steps = session.run([cost, learning_rate, global_step], feed_dict=val_feed)
            val_ler = 1
        print("val get to next")
        log = "Epoch {}/{}, steps = {}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}s, learning_rate = {}"
        print(
            log.format(curr_epoch + 1, 10000, steps, train_cost, train_ler, val_cost, val_ler, time.time() - start,
                       lr))
        if curr_epoch % 5  == 0:
            do_report(session)
            
