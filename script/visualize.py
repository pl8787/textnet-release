#-*-coding:utf8-*-
import json
import Image 
import numpy as np

def load_tensor(json_root):
    shape = json_root['data']['shape']
    len = shape[0]*shape[1]*shape[2]*shape[3]
    t = np.zeros(len)
    for i in range(len):
        t[i] = json_root['data']['value'][i]
    t = t.reshape(shape)
    return t

def draw_image(pixels, ofile):
    img = Image.new( 'RGB', (len(pixels),len(pixels[0])), "black") # create a new black image
    ps = img.load() # create the pixel map
                                            
    for i in range(len(pixels)):    # for every pixel:
        for j in range(len(pixels[i])):
            ps[i,j] = (pixels[i][j], pixels[i][j], pixels[i][j]) # set the colour accordingly

    print ofile                                          
    img.save(ofile)

def draw_01_image(pixels, ofile):
    for i in range(len(pixels)):
        for j in range(len(pixels[i])):
            pixels[i][j] *= 255
    draw_image(pixels, ofile)

# -1到1的real值的图
def draw_neg11_real_image(pixels, ofile):
    for i in range(len(pixels)):
        for j in range(len(pixels[i])):
            pixels[i][j] += 1
            pixels[i][j] *= 254/2
    draw_image(pixels, ofile)

def output_matrix(mat, ofile):
    fo = open(ofile, 'w')
    for l in mat:
        for i in l:
            fo.write(str(i) + ' ')
        fo.write('\n')
    fo.close()

def visualize():
    test_iter = 10000 
    test_file = './test.' + str(test_iter)
    batch = json.loads(open(test_file).read())[0]

    xor_sim = np.swapaxes(load_tensor(batch['xor_similarity_swap']), 1, 3)
    print 'xor_sim_shape:', xor_sim.shape
    
    rnn_sim = np.swapaxes(load_tensor(batch['sim_recurrent']),1, 3)
    print 'rnn_sim.shape:', xor_sim.shape


    output_matrix(rnn_sim[42][0].tolist(), './tmp')
    exit(0)
    img_prefix = './img.{0}/'.format(str(test_iter))
    for i in range(rnn_sim.shape[0]):
        draw_01_image(xor_sim[i][0].tolist(), img_prefix + str(i)+'.png')
    for i in range(rnn_sim.shape[0]):
        for j in range(rnn_sim.shape[1]):
            draw_neg11_real_image(rnn_sim[i][j].tolist(), img_prefix + str(i)+'_'+str(j)+'_rnn.png')

visualize()
