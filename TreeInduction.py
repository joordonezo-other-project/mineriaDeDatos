import math
data = [
    {
        'age':'<=30',
        'income': 'high',
        'student':'no',
        'credit_rating':'fair',
        'buys_computer':'no'
    },
    {
        'age':'<=30',
        'income': 'high',
        'student':'no',
        'credit_rating':'excellent',
        'buys_computer':'no'
    },
    {
        'age':'>30<=40',
        'income': 'high',
        'student':'no',
        'credit_rating':'fair',
        'buys_computer':'yes'
    },
    {
        'age':'>40',
        'income': 'medium',
        'student':'no',
        'credit_rating':'fair',
        'buys_computer':'yes'
    },
    {
        'age':'>40',
        'income': 'low',
        'student':'yes',
        'credit_rating':'fair',
        'buys_computer':'yes'
    },
    {
        'age':'>40',
        'income': 'low',
        'student':'yes',
        'credit_rating':'excellent',
        'buys_computer':'no'
    },
    {
        'age':'>30<=40',
        'income': 'low',
        'student':'yes',
        'credit_rating':'excellent',
        'buys_computer':'yes'
    },
    {
        'age':'<=30',
        'income': 'medium',
        'student':'no',
        'credit_rating':'fair',
        'buys_computer':'no'
    },
    {
        'age':'<=30',
        'income': 'low',
        'student':'yes',
        'credit_rating':'fair',
        'buys_computer':'yes'
    },
    {
        'age':'>40',
        'income': 'medium',
        'student':'yes',
        'credit_rating':'fair',
        'buys_computer':'yes'
    },
     {
        'age':'<=30',
        'income': 'medium',
        'student':'yes',
        'credit_rating':'excellent',
        'buys_computer':'yes'
    },
    {
        'age':'>30<=40',
        'income': 'medium',
        'student':'no',
        'credit_rating':'excellent',
        'buys_computer':'yes'
    },
    {
        'age':'>30<=40',
        'income': 'high',
        'student':'yes',
        'credit_rating':'fair',
        'buys_computer':'yes'
    },
    {
        'age':'>40',
        'income': 'medium',
        'student':'no',
        'credit_rating':'excellent',
        'buys_computer':'no'
    }
]

columnBinary = 'buys_computer'

def Gain(type):
    return infoD()-infoTypeD(type)

def I(num1,num2):
    total = (num1+num2)
    resDiv1 =(num1/total)
    resDiv2 =(num2/total)
    log1 = 0
    log2 = 0
    
    if (resDiv1 > 0):
        log1 = math.log(resDiv1,2)
        
    if (resDiv2 > 0):
        log2 = math.log(resDiv2,2)

    return (-resDiv1*log1-resDiv2*log2)

def infoD ():
    return I(9,5)

def infoTypeD(type):
    counters = getProbabilidata(type)
    totalData = len(data)
    response = 0
    for _, valor in counters.items():
        yes =valor['yes']
        no = valor['no']
        totalCount = yes + no
        response += ((totalCount/totalData)*I(yes,no))
    return response

def getProbabilidata(column):
    diferentsValues = []
    for value in data:
        if(value[column] not in diferentsValues):
           diferentsValues.append(value[column])
    
    counters = {}
    for category in diferentsValues:
        counters[category]={
            "yes":0,
            "no":0
        }

    for value in data:
        if(value[columnBinary] == 'yes'):
            counters[value[column]]['yes'] += 1
        elif(value[columnBinary] == 'no'):
            counters[value[column]]['no'] += 1

    return counters
        

print("Age: " + str(Gain('age')))
print("Income: " + str(Gain('income')))
print("Student: " + str(Gain('student')))
print("Credit Rating: " + str(Gain('credit_rating')))