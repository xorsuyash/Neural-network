import math 
def sigmoid(z):
    y=1.0/(1+math.exp(-z))
    return y 

def activate(inputs,weights):
    #dot product
    z=0 
    for x,w in zip(inputs, weights):
        z+=x*w 

    #perform the activation 
    return sigmoid(z)



if __name__ == "__main__":
    inputs=[.5,.6,.2]
    weights=[.4,.7,.2]
    output = activate(inputs , weights)
    print(output)



#weights[[w11,w12,w13],[w21,w22,w23]]
