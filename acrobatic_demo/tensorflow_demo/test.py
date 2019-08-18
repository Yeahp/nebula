class Test(object):
	def __init__(self):
		self.__num = 100
	#@getNum.setter #等同步于 porperty(setNum,getNum)
	def setNum(self,num): #将self.__num的属性封装。
		self.__num = num

	#@property #等于getNum = porperty(getNum) 默认的是getter方法。
	def getNum(self): #获取__num的值。
		return self.__num
	num = property(getNum,setNum) #使用关键字porperty将getNum和setNum方法打包使用，并将引用赋予属性num。

t = Test()
#print(t.__num) #将会出错，表示输出私有属性，外部无法使用。
t.__num = 200  #这里将会理解为添加属性 __num = 200,而不是重新赋值私有属性。
print(t.__num) #这里输出的200是定义的属性__num，而不是self.__num。
t.setNum(100) #通过set方法将私有属性重新赋值。
print(t.getNum()) #通过get方法获取__num的值。
#print(_Test__num) #私有属性其实是系统再私有属性前加上了一个_Test，就是一个下划线加类名。

t.num = 300 #调用类属性num,并重新赋值，porperty会自动检测set方法和get方法，并将引用赋值给set方法。
print(t.getNum()) #输出类属性，并会自己检测使用get方法进行输出