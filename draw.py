from matplotlib import pyplot as plt
import numpy as np
import glob
import os

def get_word_containing(string,char):
    words = [word[5:] for word in string.split('') if char in word]
    return words


#画LOSS

path1=r'./uformer512_log\log\Uformer_uniformer16_0716_uniformer_lepe/'
files = glob.glob(os.path.join(path1,'*.txt'))
f=open(files[0])
epochs=[]
loss=[]
word='Loss: '
for line in f:
    if word in line:
        half=line.split(word)
        num=half[-1].split()
        loss.append(float(num[0]))

        half1=line.split('Epoch: ')
        epoch=half1[1].split()
        epochs.append(int(epoch[0]))


        # epoch_index=line.find('Epoch')
        # loss.append()


path2=r'./uformer512_log\log\Uformer_uniformer16_0616_rice1_uniformernew_first1.5_swin1_seed/'
files1 = glob.glob(os.path.join(path2,'*.txt'))
f=open(files1[0])
epochs2=[]
loss2=[]
word='Loss: '
for line in f:
    if word in line:
        half=line.split(word)
        num=half[-1].split()
        loss2.append(float(num[0]))

        half1=line.split('Epoch: ')
        epoch=half1[1].split()
        epochs2.append(int(epoch[0]))


        # epoch_index=line.find('Epoch')
        # loss.append()



plt.plot(np.array(epochs),np.array(loss),label='lepe_uni')
plt.plot(np.array(epochs2),np.array(loss2),label='rpe_uni')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.legend(loc='best')

plt.show()



#画PSNR

# path1=r'./uformer512_log\log\Uformer_uniformer16_0716_uniformer_lepe/'
# files = glob.glob(os.path.join(path1,'*.txt'))
# f=open(files[0])
# epochs=[]
# loss=[]
# word='it 159'
# for line in f:
#     if word in line:
#         half=line.split(word)
#         num=half[-1].split('PSNR SIDD: ')
#         num=num[1].split()
#         loss.append(float(num[0]))
#
#         half1=line.split('Ep ')
#         epoch=half1[1].split()
#         epochs.append(int(epoch[0]))
#
#
#         # epoch_index=line.find('Epoch')
#         # loss.append()
#
#
# path2=r'./uformer512_log\log\Uformer_uniformer16_0616_rice1_uniformernew_first1.5_swin1_seed/'
# files1 = glob.glob(os.path.join(path2,'*.txt'))
# f=open(files1[0])
# epochs2=[]
# loss2=[]
# word='it 159'
# for line in f:
#     if word in line:
#         half = line.split(word)
#         num = half[-1].split('PSNR SIDD: ')
#         num = num[1].split()
#         loss2.append(float(num[0]))
#
#         half1 = line.split('Ep ')
#         epoch = half1[1].split()
#         epochs2.append(int(epoch[0]))
#
#
#         # epoch_index=line.find('Epoch')
#         # loss.append()
#
#
#
# plt.plot(np.array(epochs),np.array(loss),label='lepe_uni')
# plt.plot(np.array(epochs2),np.array(loss2),label='rpe')
# plt.xlabel('epochs')
# plt.ylabel('Loss')
# plt.legend(loc='best')
#
# plt.show()

print('hh')


