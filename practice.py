import numpy as np 
class MLP:

    def __init__(self,num_inputs,num_hidden,num_output):
        self.num_inputs=num_inputs 
        self.num_hidden=num_hidden
        self.num_output=num_output
        layers=[num_inputs]+num_hidden+[num_output]

        #random initialization of weights 
        self.weights=[]
        for i in range (len(layers)-1):
            w=np.random.rand(layers[i],layers[i+1])
            self.weights.append(w)

    #Activation function 
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))

    def forward_prop(self,inputs):
        activations=inputs 
        for w in self.weights:
            net_input=np.dot(activations,w)
            activations=self.sigmoid(net_input)

        return activations 

if __name__=="__main__":
    mlp=MLP(2,[3,5],2)
    input=np.random.rand(mlp.num_inputs)
    output=mlp.forward_prop(input)

    print(f"input is {input}")
    print(f"output of the following network is {output}")


