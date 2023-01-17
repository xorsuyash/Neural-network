import numpy as np
#save activations and derivatives 
#implement backpropagation 
#implement gradient descent 
#implement train 
#train our net with some dummy dataset 
#make some predictions 
class MlP :
    def __init__(self,num_inputs,num_hidden,num_outputs ):
        self.num_inputs=num_inputs 
        self.num_hidden=num_hidden
        self.num_outputs=num_outputs    
        layers=[self.num_inputs]+self.num_hidden+[self.num_outputs ]
    

        #initiate random weights  
        self.weights = []
        for i in range(len(layers)-1):
             w= np.random.rand((layers[i],layers[i+1]))
             self.weights.append(w)
        #save activations 
        activations=[]
        for i in range(len(layers)):
            a=np.zeros(layers[i])
            activations.append(a)
        self.activations=activations    

        #saving derivatives 
        derivatives=[]
        for i in range(len(layers)-1):
            d=np.zeros((layers[i],layers[i+1]))
            derivatives.append(d)
        self.derivatives=derivatives   

           
    #forward propagation

    def forward_prop(self, inputs):
        activations = inputs 
        self.activations[0]=inputs

        for i,w in enumerate(self.weights):
            #calculate net inputs
            net_inputs = np.dot(activations, w) 
            #calculate the actiavtions 

            activations=self._sigmoid(net_inputs)
            self.activations[i+1]=activations 

        return activations 
    #back_popagations 
    def back_propagation(self,error):
        for i in reversed(range(len(self.derivatives))):
            activations=self.activations[i+1]
            delta = error*self.sigmoid_derivative(activations)
            delta_reshaped=
            current_activations=self.activations[i]
            curent_activations_reshaped=current_activations.reshapecurent_activations_reshaped=(current_activations.shape[0],-1)
            
            self.derivatives[i]=np.dot(curent_activations_reshaped,delta)




    def sigmoid_derivative(self,x):
        return x*(1-x)

    
    #sigmoid activation function 
    
    def _sigmoid(self,x):
        return 1/(1+np.exp(-x))
if __name__ == "__main__":
    #create a mlp
    mlp=MlP(3,[3,5],2) 
    #create inputs 
    inputs=np.random.rand(mlp.num_inputs)
    #perform forward prop 
    outputs=mlp.forward_prop(inputs)
    #print the results 
    print(f"The network input is : {inputs}")
    print(f"The network output is : {outputs}")




