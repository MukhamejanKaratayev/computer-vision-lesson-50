

class Dog:
    
    def __init__(self, name: str = 'Dog'):
        self.name = name
    
    def __call__(self, x1: int = 12) -> str:
        return f'{self.name} is {x1} years old'

    def bark(self):
        print(f'{self.name} is barking')

    def move(self):
        print('moving')
    
    