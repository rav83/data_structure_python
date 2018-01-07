##############################################
#Implement a linked list

class LinkedListNode(object):
    
    def __init__(self,value):
        
        self.value = value
        self.nextnode = None
 
##############################################
#Implement a doubly linked list
class DoublyLinkedListNode(object):
    
    def __init__(self,value):
        self.value = value
        self.nextnode = None
        self.prevnode = None

##############################################
#Linked List Reversal
#return first node of reversed linked list
#parameter is head node
def ll_reversal(n):
    
    previous = None
    curr_node = n
    temp = None
    
    while curr_node:
        temp = curr_node.nextnode
        curr_node.nextnode = previous
        
        previous = curr_node
        curr_node = temp
        
    return previous

#Test
# Create a list of 4 nodes
a = LinkedListNode(1)
b = LinkedListNode(2)
c = LinkedListNode(3)
d = LinkedListNode(4)

# Set up order a,b,c,d with values 1,2,3,4
a.nextnode = b
b.nextnode = c
c.nextnode = d

ll_reversal(a)

print (b.nextnode.value)
print (c.nextnode.value)
print (d.nextnode.value)
 
##############################################

#Singly Linked List Cycle Check

def cycle_check(node):
    
    # Begin both markers at the first node
    marker1 = node
    marker2 = node
    
    # Go until end of list
    while marker2 != None and marker2.nextnode != None:
        marker1 = marker1.nextnode
        marker2 = marker2.nextnode.nextnode
        
        # Check if the markers have crossed
        if marker1 == marker2:
            return True
        
    # Case where marker ahead reaches the end of the list
    return False
        

 
##############################################
#Linked List Nth to Last Node

def nth_to_last_node(n, head):

    left_pointer  = head
    right_pointer = head

    # Set right pointer at n nodes away from head
    for i in range(n-1):
        
        # Check for edge case of not having enough nodes!
        if not right_pointer.nextnode:
            raise LookupError('Error: n is larger than the linked list.')

        # Otherwise, we can set the block
        right_pointer = right_pointer.nextnode

    # Move the block down the linked list
    while right_pointer.nextnode:
        left_pointer  = left_pointer.nextnode
        right_pointer = right_pointer.nextnode

    # Now return left pointer, its at the nth to last element!
    return left_pointer
 
##############################################

# A complete working Python program to demonstrate all
# insertion methods of linked list
'''
In this post, methods to insert a new node in linked list are discussed. A node can be added in three ways
1) At the front of the linked list
2) After a given node.
3) At the end of the linked list.
'''

# Node class
class Node:
 
    # Function to initialise the node object
    def __init__(self, data):
        self.data = data  # Assign data
        self.next = None  # Initialize next as null
 
 
# Linked List class contains a Node object
class LinkedList:
 
    # Function to initialize head
    def __init__(self):
        self.head = None
        
    # This function is in LinkedList class
    # 1) Function to insert a new node at the beginning
    def push(self, new_data):

        # 1 & 2: Allocate the Node &
        #        Put in the data
        new_node = Node(new_data)

        # 3. Make next of new Node as head
        new_node.next = self.head

        # 4. Move the head to point to new Node 
        self.head = new_node 

    # This function is in LinkedList class.
    # 2) Inserts a new node after the given prev_node. This method is 
    # defined inside LinkedList class shown above
    def insertAfter(self, prev_node, new_data):

        # 1. check if the given prev_node exists
        if prev_node is None:
            print ("The given previous node must inLinkedList.")
            return

        #  2. Create new node &
        #  3. Put in the data
        new_node = Node(new_data)

        # 4. Make next of new Node as next of prev_node 
        new_node.next = prev_node.next

        # 5. make next of prev_node as new_node 
        prev_node.next = new_node
        
        
    # This function is defined in Linked List class
    # 3) Appends a new node at the end.  This method is
    #  defined inside LinkedList class shown above */
    def append(self, new_data):

        # 1. Create a new node
        # 2. Put in the data
        # 3. Set next as None
        new_node = Node(new_data)

        # 4. If the Linked List is empty, then make the
        #    new node as head
        if self.head is None:
            self.head = new_node
            return

        # 5. Else traverse till the last node
        last = self.head
        while (last.next):
            last = last.next

        # 6. Change the next of last node
        last.next =  new_node

 
##############################################
# Given a reference to the head of a list and a key,
# delete the first occurence of key in linked list
    def deleteNode(self, key):
         
        # Store head node
        temp = self.head
 
        # If head node itself holds the key to be deleted
        if (temp is not None):
            if (temp.data == key):
                self.head = temp.next
                temp = None
                return
 
        # Search for the key to be deleted, keep track of the
        # previous node as we need to change 'prev.next'
        while(temp is not None):
            if temp.data == key:
                break
            prev = temp
            temp = temp.next
 
        # if key was not present in linked list
        if(temp == None):
            return
 
        # Unlink the node from linked list
        prev.next = temp.next
 
        temp = None
 


 
##############################################
    # Utility function to print the linked LinkedList
    def printList(self):
        temp = self.head
        while(temp):
            print (temp.data),
            temp = temp.next



 
##############################################
#Implement an algorithm to delete a node in the middle of a single linked list, given only access to that node.
'''
This is different from above becasue we do not have access to previous node.
==> The solution to this is to simply copy the data from the next node into this node and then delete the next node.
This problem can not be solved if the node to be deleted is the last node
'''
def deleteNode(self, node):
    
    if node == None or node.next == None:
        return False
    
    node.value = node.nextnode.value
    node.nextnode = node.nextnode.nextnode
    
    return True
    
 
##############################################

#Compare two strings represented as linked lists
#Write a function compare() that works similar to strcmp(), 
#i.e., it returns 0 if both strings are same, 
#1 if first linked list is lexicographically greater, and 
#-1 if second string is lexicographically greater.


# A linked list node structure
class Node:
 
    # Constructor to create a new node
    def __init__(self, key):
        self.c = key ; 
        self.next = None
 
def compare(list1, list2):
     
    # Traverse both lists. Stop when either end of linked 
    # list is reached or current characters don't watch
    while(list1 and list2 and list1.c == list2.c):
        list1 = list1.next
        list2 = list2.next
 
    # If both lists are not empty, compare mismatching
    # characters 
    if(list1 and list2):
        return 1 if list1.c > list2.c else -1
 
    # If either of the two lists has reached end
    if (list1 and not list2):
        return 1
 
    if (list2 and not list1):
        return -1
    return 0
 
 
##############################################

# Python program to add two numbers represented by linked list
 
# Node class
class Node:
 
    # Constructor to initialize the node object
    def __init__(self, data):
        self.data = data
        self.next = None
 
class LinkedList:
 
    # Function to initialize head
    def __init__(self):
        self.head = None
 
    # Function to insert a new node at the beginning
    def push(self, new_data):
        new_node = Node(new_data)
        new_node.next = self.head
        self.head = new_node
 
    # Add contents of two linked lists and return the head
    # node of resultant list
    def addTwoLists(self, first, second):
        prev = None
        temp = None
        carry = 0
 
        # While both list exists
        while(first is not None or second is not None):
 
            # Calculate the value of next digit in
            # resultant list
            # The next digit is sum of following things
            # (i) Carry
            # (ii) Next digit of first list (if ther is a
            # next digit)
            # (iii) Next digit of second list ( if there
            # is a next digit)
            fdata = 0 if first is None else first.data
            sdata = 0 if second is None else second.data
            Sum = carry + fdata + sdata
 
            # update carry for next calculation
            carry = 1 if Sum >= 10 else 0
 
            # update sum if it is greater than 10
            Sum = Sum if Sum < 10 else Sum % 10
 
            # Create a new node with sum as data
            temp = Node(Sum)
 
            # if this is the first node then set it as head
            # of resultant list
            if self.head is None:
                self.head = temp
            else :
                prev.next = temp 
 
            # Set prev for next insertion
            prev = temp
 
            # Move first and second pointers to next nodes
            if first is not None:
                first = first.next
            if second is not None:
                second = second.next
 
        if carry > 0:
            temp.next = Node(carry)
 
    # Utility function to print the linked LinkedList
    def printList(self):
        temp = self.head
        while(temp):
            print temp.data,
            temp = temp.next
 
# Driver program to test above function
first = LinkedList()
second = LinkedList()
 
# Create first list
first.push(6)
first.push(4)
first.push(9)
first.push(5)
first.push(7)
print "First List is ",
first.printList()
 
# Create second list
second.push(4)
second.push(8)
print "\nSecond List is ",
second.printList()
 
# Add the two lists and see result
res = LinkedList()
res.addTwoLists(first.head, second.head)
print "\nResultant list is ",
res.printList()

 
##############################################

#Reverse a Linked List in groups of given size
#Example:
# Inputs:  1->2->3->4->5->6->7->8->NULL and k = 3 
# Output:  3->2->1->6->5->4->8->7->NULL. 
# 
# Inputs:   1->2->3->4->5->6->7->8->NULL and k = 5
# Output:  5->4->3->2->1->8->7->6->NULL. 


# Node class 
class Node:
 
    # Constructor to initialize the node object
    def __init__(self, data):
        self.data = data
        self.next = None
 
class LinkedList:
 
    # Function to initialize head
    def __init__(self):
        self.head = None
 
    def reverse(self, head, k):
        current = head 
        next  = None
        prev = None
        count = 0
         
        # Reverse first k nodes of the linked list
        while(current is not None and count < k):
            next = current.next
            current.next = prev
            prev = current
            current = next
            count += 1
 
        # next is now a pointer to (k+1)th node
        # recursively call for the list starting
        # from current . And make rest of the list as
        # next of first node
        if next is not None:
            head.next = self.reverse(next, k)
 
        # prev is new head of the input list
        return prev
 
    # Function to insert a new node at the beginning
    def push(self, new_data):
        new_node = Node(new_data)
        new_node.next = self.head
        self.head = new_node
 
    # Utility function to print the linked LinkedList
    def printList(self):
        temp = self.head
        while(temp):
            print temp.data,
            temp = temp.next
 
 
##############################################
#Union and Intersection of two sorted arrays
#Union
#Time Complexity: O(m+n)

def printUnion(arr1, arr2, m, n):
    i,j = 0,0
    while i < m and j < n:
        if arr1[i] < arr2[j]:
            print(arr1[i])
            i += 1
        elif arr2[j] < arr1[i]:
            print(arr2[j])
            j+= 1
        else:
            print(arr2[j])
            j += 1
            i += 1
 
    # Print remaining elements of the larger array
    while i < m:
        print(arr1[i])
        i += 1
 
    while j < n:
        print(arr2[j])
        j += 1
 
# Driver program to test above function
arr1 = [1, 2, 4, 5, 6]
arr2 = [2, 3, 5, 7]
m = len(arr1)
n = len(arr2)
printUnion(arr1, arr2, m, n)
 
 
 
 #Intersection
 #Time Complexity: O(m+n)
 
 def printIntersection(arr1, arr2, m, n):
    i,j = 0,0
    while i < m and j < n:
        if arr1[i] < arr2[j]:
            i += 1
        elif arr2[j] < arr1[i]:
            j+= 1
        else:
            print(arr2[j])
            j += 1
            i += 1
            
            
 


 
##############################################
# Python program to detect and remove loop in linked list
 
# Node class 
class Node:
 
    # Constructor to initialize the node object
    def __init__(self, data):
        self.data = data
        self.next = None
 
class LinkedList:
 
    # Function to initialize head
    def __init__(self):
        self.head = None
 
    def detectAndRemoveLoop(self):
        slow_p = fast_p = self.head
        while(slow_p and fast_p and fast_p.next):
            slow_p = slow_p.next
            fast_p = fast_p.next.next
         
            # If slow_p and fast_p meet at some poin
            # then there is a loop
            if slow_p == fast_p:
                self.removeLoop(slow_p)
                 
                # Return 1 to indicate that loop if found
                return 1
 
        # Return 0 to indicate that there is no loop
        return 0
 
    # Function to remove loop
    # loop node-> Pointer to one of the loop nodes
    # head --> Pointer to the start node of the
    # linked list
    def removeLoop(self, loop_node):
         
        # Set a pointer to the beginning of the linked 
        # list and move it one by one to find the first
        # node which is part of the linked list
        ptr1 = self.head
        while(1):
            # Now start a pointer from loop_node and check
            # if it ever reaches ptr2
            ptr2 = loop_node
            while(ptr2.next!= loop_node and ptr2.next !=ptr1):
                ptr2 = ptr2.next
             
            # If ptr2 reached ptr1 then there is a loop.
            # So break the loop
            if ptr2.next == ptr1 : 
                break
             
            ptr1 = ptr1.next
         
        # After the end of loop ptr2 is the lsat node of 
        # the loop. So make next of ptr2 as NULL
        ptr2.next = None
    # Function to insert a new node at the beginning
    def push(self, new_data):
        new_node = Node(new_data)
        new_node.next = self.head
        self.head = new_node
 
    # Utility function to prit the linked LinkedList
    def printList(self):
        temp = self.head
        while(temp):
            print temp.data,
            temp = temp.next
 


 
##############################################



 
