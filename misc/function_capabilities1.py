from math import exp 

#function as arguments 
def apply(lists, function):
    """apply function to the lists and return the values
    lists: input list 
    function: operation to be performed
    """

    output = []
    [output.append(function(lists[i])) for i in range(len(lists))] 
    #print(v)
    
    return output, function


#function as elements of list 
def applyFunctionList(lists, values):
    """apply functions arranged in list to the values passed 
    lists: this contains the functions arranged in list 
    values: the input on whic the function list is to be applied 
    """
   
    output = [i(values) for i in lists]
   
    return output 


if __name__ == "__main__":
    L = [-1,-2,-3,-4,-5]
    FL = [abs, exp, int] 
    num = -2 
    print('values before operation: ',L)
    value, function = apply(L,abs) 
    print('values after operation ', str(function), ' : ',value)
    print('----------------------')
    print('function list is: ', FL)
    value1 = applyFunctionList(FL, num)
    print('function list applied on ', num, ' : ',value1)


