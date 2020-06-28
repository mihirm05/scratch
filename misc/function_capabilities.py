def apply(lists, function):
    """apply function to the lists and return the values
    lists: input list 
    function: operation to be performed
    """

    output = []
    [output.append(function(lists[i])) for i in range(len(lists))] 
    #print(v)
    
    return output, function


if __name__ == "__main__":
    L = [-1,-2,-3,-4,-5]
    print('values before operation: ',L)
    value, function = apply(L,abs) 
    print('values after operation ', str(function), ' : ',value)
