

A = 'kakao'
B = 'chocolette'
C = 'milk'

testTuple = (A, B)

print(testTuple)
print()

tempList = list(testTuple)
tempList.append(C)
testTuple = tuple(tempList)

print(testTuple)
print()
