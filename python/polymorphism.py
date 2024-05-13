# This code was generated by Google Gemini

class Animal:
  def make_sound(self):
    print("Generic animal sound")

class Dog(Animal):
  def make_sound(self):
    print("Woof!")

class Cat(Animal):
  def make_sound(self):

    print("Meow!")

def speak(animal):
  animal.make_sound()

dog = Dog()
cat = Cat()

speak(dog)  # Output: Woof!
speak(cat)  # Output: Meow!

print(len("Hello"))  # Output: 5
print(len([1, 2, 3]))  # Output: 3
