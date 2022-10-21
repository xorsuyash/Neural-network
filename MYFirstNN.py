import numpy as suyash
inputs=[1,2,3,2.5]
weights=[[0.2,0.3,0.4,0.6],
        [0.5,-0.91,0.26,-0.5],
        [-0.26,-0.27,0.17,0.87]]
biases=[2,3,0.5]
layers_output=[]
for neuron_weight , neuron_bias in zip(weights,biases):
    neuron_output=0
    for n_inputs , n_weights in zip(inputs , neuron_weight):
        neuron_output+=n_inputs*weights
    neuron_output+=neuron_bias
    layers_output.append(neuron_output)
print(layers_output)

#we can also do the following operation by using numpy 

trans_weights=suyash.transpose(weights)
output=suyash.dot(inputs,trans_weights)
layers_outputnew=[]
for nwb,n_b in zip(output , biases):
    layers_outputnew.append(nwb+n_b)
print(layers_outputnew)

