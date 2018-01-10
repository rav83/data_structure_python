##############################################

#Implement Stack

class stack(object):
    
    def __init__(self):
        self.items = []
        
    def size(self):
        return len(self.items)
    
    def isEmpty(self):
        return self.items == []
    
    def push(self,n):
        self.items.append(n)
        
    def pop(self):
        return self.items.pop()
        
    def peek(self):
        return self.items[len(self.items)-1]


 
##############################################

#Implement Queue

class queue(object):
    
    def __init__(self):
        self.items = []
        
    def size(self):
        return len(self.items)
    
    def isEmpty(self):
        return self.items == []
    
    def enqueue(self,n):
        self.items.insert(0,n)
        
    def dequeue(self):
        return self.items.pop()
 
##############################################

#1. Queue Implementation using Linked List

class Node(object):
    
    def __init__(self):
        self.next = None
        self.data = None
        

class queue_ll(object):
    
    def __init__(self):
        self.first = None
        self.last = None
        self.size = 0
        
    def size(self):
        return self.size
        
    def enqueue(self,data):
        
        new_node = Node(data)
        
        #if empty
        if self.first is None and self.last is None:
            self.first = new_node
            self.last = new_node
        else:
            self.last.next = new_node
            self.last = new_node #mark last node as the new one
            
        size += 1
    
    def dequeue(self):
        
        if self.first is None:
            return None
        else:
            temp = self.first
            
            #check if there is only one node, need to update last
            if self.first.next is None:
                self.first = None
                self.last = None
            else:
                self.first = self.first.next
                
            size -= 1
            
            return temp
 
##############################################
#Balanced Parentheses Check
# {([])} --> true, 
#{([}]) --> false


def balance_check(s):
    
    if len(s) == 0:
        return
    
    #length should be even for all pairs to be present
    if len(s)%2 != 0:
        return False
    
    #to check the opening ones
    #To Do: Use Set instead of list 
    # opening_brackets = set('[{(')
    opening_brackets = '[{('
    
    #pairs 
    #To Do: Use Set instead of list 
    #Sets are significantly faster when it comes to determining 
    #if an object is present in the set (as in x in s), 
    #but are slower than lists when it comes to iterating over their contents.
    # it should be ---> pairs = set([('[',']'),('{','}'),('(',')')])

    pairs = [('[',']'),('{','}'),('(',')')]
    
    #stack to store opening ones
    stack = []
    
    for i in s:
        if i in opening_brackets:
            stack.append(i)
        else:
            
            opening_found = stack.pop()
            
            if (opening_found,i) not in pairs:
                return False
            
    if len(stack) != 0:
        return False
    
    return True


balance_check('({{}()()})')

##############################################


#Implement a Queue - Using Two Stacks

#remember to use class because functions will be defined

class stacks_to_queue(object):
    
    def __init__(self):
        #use self.. very important
        self.stack1 = []
        self.stack2 = []
        
    def enqueue(self,element):
        self.stack1.append(element)
        
    def dequeue(self):
        if len(self.stack2)== 0:
            while self.stack1:
                # Add the elements to the outstack to reverse the order when called
                self.stack2.append(self.stack1.pop())
                
        return self.stack2.pop()
            
    
 
##############################################


# Chess Knight Problem – Find Shortest path from source to destination

class Node(object):
    
    def __init__(self):
        self.x = 0
        self.y = 0
        self.dist = 0
    
def valid(x,y):
    if x < 0 or y < 0 or x > 8 or y > 8:
        return False
    
    return True


def find_shortest(source_x, source_y ,dest_x, dest_y):
    
    n = Node()
    n.x = source_x
    n.y = source_y
    n.dist = 0
    
    
    queue = [n]
    
    visited = [(n.x,n.y)]
    
    
    while (queue != []):
        
        curr = queue.pop(0)
        
        
        for i,j in [(1,2),(1,-2),(2,1),(2,-1),(-1,2),(-1,-2),(-2,1),(-2,-1)]:
            
            new_x = curr.x + i
            new_y = curr.y + j
            
            if (new_x == dest_x and new_y == dest_y):
                return curr.dist + 1
            
            if valid(new_x, new_y)  and (new_x, new_y) not in visited:
                
                visited.append((new_x, new_y))
                
                new_node = Node()
                new_node.x = new_x
                new_node.y = new_y
                new_node.dist = curr.dist + 1
                
                queue.append(new_node)


find_shortest(0,7,7,0)

 #Extend this to add path as well
##############################################



 
##############################################

