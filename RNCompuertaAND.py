entradas = [1,1,1]
pesos = [0.1,0.2,-0.3]

def activationFuction(value):
    if(value < 0.0):
        return 0
    elif(value >=0.0):
        return 1
    
def neuron(entradas,pesos):
    result = []
    for i in range(len(entradas)):
        result.append((entradas[i]*pesos[i]))

    return activationFuction(sum(result))

output = neuron(entradas, pesos)
print(output)