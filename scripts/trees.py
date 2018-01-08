##############################################
#Nodes and References Implementation of a Tree
class BinaryTree(object):
    def __init__(self,rootObj):
        self.key = rootObj
        self.leftChild = None
        self.rightChild = None

    def insertLeft(self,newNode):
        if self.leftChild == None:
            self.leftChild = BinaryTree(newNode)
        else:
            t = BinaryTree(newNode)
            t.leftChild = self.leftChild
            self.leftChild = t

    def insertRight(self,newNode):
        if self.rightChild == None:
            self.rightChild = BinaryTree(newNode)
        else:
            t = BinaryTree(newNode)
            t.rightChild = self.rightChild
            self.rightChild = t


    def getRightChild(self):
        return self.rightChild

    def getLeftChild(self):
        return self.leftChild

    def setRootVal(self,obj):
        self.key = obj

    def getRootVal(self):

##############################################
#Tree Representation Implementation (Lists)
def BinaryTree(r):
    return [r, [], []]

def insertLeft(root,newBranch):
    t = root.pop(1)
    if len(t) > 1:
        root.insert(1,[newBranch,t,[]])
    else:
        root.insert(1,[newBranch, [], []])
    return root

def insertRight(root,newBranch):
    t = root.pop(2)
    if len(t) > 1:
        root.insert(2,[newBranch,[],t])
    else:
        root.insert(2,[newBranch,[],[]])
    return root

def getRootVal(root):
    return root[0]

def setRootVal(root,newVal):
    root[0] = newVal

def getLeftChild(root):
    return root[1]

def getRightChild(root):
    return root[2]


##############################################
#Binary Heap Implementation
class BinHeap:
    def __init__(self):
        self.heapList = [0]
        self.currentSize = 0


    def percUp(self,i):
        
        while i // 2 > 0:
            
            if self.heapList[i] < self.heapList[i // 2]:
                
            
                tmp = self.heapList[i // 2]
                self.heapList[i // 2] = self.heapList[i]
                self.heapList[i] = tmp
            i = i // 2

    def insert(self,k):
        
        self.heapList.append(k)
        self.currentSize = self.currentSize + 1
        self.percUp(self.currentSize)

    def percDown(self,i):
        
        while (i * 2) <= self.currentSize:
            
            mc = self.minChild(i)
            if self.heapList[i] > self.heapList[mc]:
                
                tmp = self.heapList[i]
                self.heapList[i] = self.heapList[mc]
                self.heapList[mc] = tmp
            i = mc

    def minChild(self,i):
        
        if i * 2 + 1 > self.currentSize:
            
            return i * 2
        else:
            
            if self.heapList[i*2] < self.heapList[i*2+1]:
                return i * 2
            else:
                return i * 2 + 1

    def delMin(self):
        retval = self.heapList[1]
        self.heapList[1] = self.heapList[self.currentSize]
        self.currentSize = self.currentSize - 1
        self.heapList.pop()
        self.percDown(1)
        return retval

    def buildHeap(self,alist):
        i = len(alist) // 2
        self.currentSize = len(alist)
        self.heapList = [0] + alist[:]
        while (i > 0):
            self.percDown(i)
            i = i - 1


##############################################
#Binary Search Trees

class TreeNode:
    
    def __init__(self,key,val,left=None,right=None,parent=None):
        self.key = key
        self.payload = val
        self.leftChild = left
        self.rightChild = right
        self.parent = parent

    def hasLeftChild(self):
        return self.leftChild

    def hasRightChild(self):
        return self.rightChild

    def isLeftChild(self):
        return self.parent and self.parent.leftChild == self

    def isRightChild(self):
        return self.parent and self.parent.rightChild == self

    def isRoot(self):
        return not self.parent

    def isLeaf(self):
        return not (self.rightChild or self.leftChild)

    def hasAnyChildren(self):
        return self.rightChild or self.leftChild

    def hasBothChildren(self):
        return self.rightChild and self.leftChild

    def replaceNodeData(self,key,value,lc,rc):
        self.key = key
        self.payload = value
        self.leftChild = lc
        self.rightChild = rc
        if self.hasLeftChild():
            self.leftChild.parent = self
        if self.hasRightChild():
            self.rightChild.parent = self


class BinarySearchTree:

    def __init__(self):
        self.root = None
        self.size = 0

    def length(self):
        return self.size

    def __len__(self):
        return self.size

    def put(self,key,val):
        if self.root:
            self._put(key,val,self.root)
        else:
            self.root = TreeNode(key,val)
        self.size = self.size + 1

    def _put(self,key,val,currentNode):
        if key < currentNode.key:
            if currentNode.hasLeftChild():
                   self._put(key,val,currentNode.leftChild)
            else:
                   currentNode.leftChild = TreeNode(key,val,parent=currentNode)
        else:
            if currentNode.hasRightChild():
                   self._put(key,val,currentNode.rightChild)
            else:
                   currentNode.rightChild = TreeNode(key,val,parent=currentNode)

    def __setitem__(self,k,v):
        self.put(k,v)

    def get(self,key):
        if self.root:
            res = self._get(key,self.root)
            if res:
                
                return res.payload
            else:
                return None
        else:
            return None

    def _get(self,key,currentNode):
        
        if not currentNode:
            return None
        elif currentNode.key == key:
            return currentNode
        elif key < currentNode.key:
            return self._get(key,currentNode.leftChild)
        else:
            return self._get(key,currentNode.rightChild)

    def __getitem__(self,key):
        return self.get(key)

    def __contains__(self,key):
        if self._get(key,self.root):
            return True
        else:
            return False

    def delete(self,key):
        
        if self.size > 1:
            
            nodeToRemove = self._get(key,self.root)
            if nodeToRemove:
                self.remove(nodeToRemove)
                self.size = self.size-1
            else:
                raise KeyError('Error, key not in tree')
        elif self.size == 1 and self.root.key == key:
            self.root = None
            self.size = self.size - 1
        else:
            raise KeyError('Error, key not in tree')

    def __delitem__(self,key):
        
        self.delete(key)

    def spliceOut(self):
        if self.isLeaf():
            if self.isLeftChild():
                
                self.parent.leftChild = None
            else:
                self.parent.rightChild = None
        elif self.hasAnyChildren():
            if self.hasLeftChild():
                
                if self.isLeftChild():
                    
                    self.parent.leftChild = self.leftChild
                else:
                    
                    self.parent.rightChild = self.leftChild
                    self.leftChild.parent = self.parent
        else:
                    
            if self.isLeftChild():
                        
                self.parent.leftChild = self.rightChild
            else:
                self.parent.rightChild = self.rightChild
                self.rightChild.parent = self.parent

    def findSuccessor(self):
        
        succ = None
        if self.hasRightChild():
            succ = self.rightChild.findMin()
        else:
            if self.parent:
                
                if self.isLeftChild():
                    
                    succ = self.parent
                else:
                    self.parent.rightChild = None
                    succ = self.parent.findSuccessor()
                    self.parent.rightChild = self
        return succ

    def findMin(self):
        
        current = self
        while current.hasLeftChild():
            current = current.leftChild
        return current

    def remove(self,currentNode):
        
        if currentNode.isLeaf(): #leaf
            if currentNode == currentNode.parent.leftChild:
                currentNode.parent.leftChild = None
            else:
                currentNode.parent.rightChild = None
        elif currentNode.hasBothChildren(): #interior
            
            succ = currentNode.findSuccessor()
            succ.spliceOut()
            currentNode.key = succ.key
            currentNode.payload = succ.payload

        else: # this node has one child
            if currentNode.hasLeftChild():
                if currentNode.isLeftChild():
                    currentNode.leftChild.parent = currentNode.parent
                    currentNode.parent.leftChild = currentNode.leftChild
                elif currentNode.isRightChild():
                    currentNode.leftChild.parent = currentNode.parent
                    currentNode.parent.rightChild = currentNode.leftChild
                else:
                
                    currentNode.replaceNodeData(currentNode.leftChild.key,
                                    currentNode.leftChild.payload,
                                    currentNode.leftChild.leftChild,
                                    currentNode.leftChild.rightChild)
            else:
                
                if currentNode.isLeftChild():
                    currentNode.rightChild.parent = currentNode.parent
                    currentNode.parent.leftChild = currentNode.rightChild
                elif currentNode.isRightChild():
                    currentNode.rightChild.parent = currentNode.parent
                    currentNode.parent.rightChild = currentNode.rightChild
                else:
                    currentNode.replaceNodeData(currentNode.rightChild.key,
                                    currentNode.rightChild.payload,
                                    currentNode.rightChild.leftChild,
                                    currentNode.rightChild.rightChild)

##############################################
#BFS
# Tree Level Order Print -  Breadth first search
#Breadth First Traversal
class Node(object):
    
    def __init__(self, val = None):
        self.right = None
        self.left = None
        self.value = val
        
def levelOrderPrint(root):
    
    if not root:
        return None
    else:
        curr_level = [root]
        
    while curr_level:
        
        node = curr_level.pop(0)
        if node.left:
            curr_level.append(node.left)
        if node.right:
            curr_level.append(node.right)
        
        print (node.value)

r = Node(0)
a = Node(1)
b = Node(2)
c = Node(3)
d = Node(4)
e = Node(5)

r.left = a
r.right = b
a.left = c
a.right = d
b.left = e
levelOrderPrint(r)
##############################################
#BFS - Breadth First Search
# print it in level order

def levelOrderPrintLevel(root):
    
    if not root:
        return None
    else:
        curr_level = [root]
    
    curr_level_count = 1
    next_level_count = 0
    
    while curr_level:
        
        node = curr_level.pop(0)
        print (node.value)
        curr_level_count -= 1
        if node.left:
            curr_level.append(node.left)
            next_level_count += 1
        if node.right:
            curr_level.append(node.right)
            next_level_count += 1
            
        
        #number of nodes in next level should now become current level count
        if curr_level_count == 0:
            temp = next_level_count
            next_level_count = curr_level_count
            curr_level_count = temp
            print ('\n')

levelOrderPrintLevel(r)

##############################################
class Node(object):
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def traverse(rootnode):
    thislevel = [rootnode]
    while thislevel:
        nextlevel = list()
        for n in thislevel:
            print (n.value)
            if n.left: nextlevel.append(n.left)
            if n.right: nextlevel.append(n.right)
        print ('\n')
        thislevel = nextlevel

##############################################

#inorder tree print

def inOrderPrint(root):
    
    if root:
        if root.left:
            inOrderPrint(root.left)
            
        print(root.value)
        
        if root.right:
            inOrderPrint(root.right)

##############################################
#print the tree - preOrder
def preOrder(node):
    if node is None:
        return;

    print(node.value);
    preOrder(node.left);
    preOrder(node.right);

##############################################
#print the tree - inOrder
def inOrder(node):
    if node is None:
        return;

    inOrder(node.left);
    print(node.value);
    inOrder(node.right);


##############################################
#Binary Search Tree Check
#Validate BST

class Node(object):
    
    def __init__(self, val = None):
        self.right = None
        self.left = None
        self.value = val
        
tree_check = []

def inOrderPrintCheck(root):

    if root:
        if root.left:
            inOrderPrintCheck(root.left)
#         print(root.value)
        tree_check.append(root.value)
        
        if root.right:
            inOrderPrintCheck(root.right)


            
def check_sort(t):
    return t == sorted(t)
    

##############################################
#Validate BST (recursive)

class Node(object):
    
    def __init__(self, val = None):
        self.right = None
        self.left = None
        self.value = val
        

        
def ValidateBST(root):
    
    return isBST(root,-100000,100000)
    
def isBST(node, min_val, max_val):
    
    if node is None:
        return True
    
    if node.value < min_val or node.value > max_val:
        return False
    
    return isBST(node.left,min_val, node.value -1) and isBST(node.right, node.value +1, max_val)


##############################################

#Trim a Binary Search Tree

class Node(object):
    
    def __init__(self, val = None):
        self.right = None
        self.left = None
        self.value = val
        


def trimBST(tree_root, minVal, maxVal):
    
    if not tree_root:
        return
    
    tree_root.left = trimBST(tree_root.left, minVal, maxVal)
    tree_root.right = trimBST(tree_root.right, minVal, maxVal)

    if minVal <= tree_root.value <= maxVal:
        return tree_root

    if tree_root.value < minVal:
        return tree.right
    
    if tree_root.value > maxVal:
        return tree.left
    

##############################################
#Size of tree
class Node(object):
    
    def __init__(self, val = None):
        self.right = None
        self.left = None
        self.value = val
        
    
def treeSize(tree_root):
    
    if not tree_root:
        return 0

    return treeSize(tree_root.left)  + 1 + treeSize(tree_root.right)



##############################################
#max/min of tree
class Node(object):
    
    def __init__(self, val = None):
        self.right = None
        self.left = None
        self.value = val
        
def maxTree(tree_root):
    
    if not tree_root:
        return float('-inf')
    
    max_v = tree_root.value
    
    left = maxTree(tree_root.left)
    right = maxTree(tree_root.right)
    
    if left >= max_v:
        max_v = left
    
    if right >= max_v:
        max_v = right
        
    return max_v



##############################################

#check depth of binary tree 
#same as below
class Node(object):
    
    def __init__(self, val = None):
        self.right = None
        self.left = None
        self.value = val

        
def heightOfTree(tree_node):
    if tree_node is None:
        return 0
    else:
        
        return 1 + max(heightOfTree(tree_node.left), heightOfTree(tree_node.right))
    
##############################################
#check if a tree is balanced
# max heightTree - min heightTree should not be greater than 1

class Node(object):
    
    def __init__(self, val = None):
        self.right = None
        self.left = None
        self.value = val
        
def maxHeightTree(tree_node):
    
    if not tree_node:
        return 0
    
    return 1 + max(maxHeightTree(tree_node.left), maxHeightTree(tree_node.right))


def minHeightTree(tree_node):
    
    if not tree_node:
        return 0
    
    return 1 + min(minHeightTree(tree_node.left), minHeightTree(tree_node.right))


def checkBalancedTree(tree_node):
    
    return maxHeightTree(tree_node) - minHeightTree(tree_node) <= 1

##############################################
#Given a sorted (increasing order) array, write an algorithm to create a binary tree with minimal height

class Node(object):
    
    def __init__(self, val = None):
        self.right = None
        self.left = None
        self.value = val

def createBSTNode(arr,start,end):
    
    if start > end:
        return None
    
    mid = (start + end) // 2
    
    n = Node(arr[mid])
    n.left = createBSTNode(arr,start,mid-1)
    n.right = createBSTNode(arr,mid+1,end)
    
    return n

def CreateBST(arr):
    return createBSTNode(arr,0,len(arr) -1)



##############################################

#Given a binary search tree, design an algorithm which creates a linked list 
# of all the nodes at each depth (eg, if you have a tree with depth D, you’ll have D linked lists).
class Node(object):
    
    def __init__(self, val = None):
        self.right = None
        self.left = None
        self.value = val

class LinkedListNode(object):
    
    def __init__(self,value = None):
        
        self.value = value
        self.nextnode = None

def BFS_ll(root):
    
    if root is None:
        return None
    
    queue = [root]
    curr_count, next_level_count = 1,0
    head_ll = None
    curr_ll = None
    arr_ll = []
    
    while len(queue)>0:
        
        curr_node = queue.pop(0)
        curr_count -= 1
        
        ll = LinkedListNode(curr_node.value)
        
        if head_ll is None:
            head_ll = ll
            curr_ll = ll
        else:
            curr_ll.nextnode = ll
            curr_ll = curr_ll.nextnode
        
        if curr_node.left:
            queue.append(curr_node.left)
            next_level_count += 1
        if curr_node.right:
            queue.append(curr_node.right)
            next_level_count += 1
            
        if curr_count == 0:
            temp = next_level_count
            next_level_count = 0
            curr_count = next_level_count
            arr_ll.append(head_ll)
            print('hi')
            head_ll = None
            curr_ll = None
    print (len(arr_ll))
##############################################

#Inorder Successor in Binary Search Tree

# Python program to find the inroder successor in a BST
 
# A binary tree node 

# 1) If right subtree of node is not NULL, then succ lies in right subtree. Do following.
# Go to right subtree and return the node with minimum key value in right subtree.
# 2) If right sbtree of node is NULL, then succ is one of the ancestors. Do following.
# Travel up using the parent pointer until you see a node which is left child of it’s parent. The parent of such a node is the succ.


class Node:
 
    # Constructor to create a new node
    def __init__(self, key):
        self.data = key 
        self.left = None
        self.right = None
 
def inOrderSuccessor(root, n):
     
    # Step 1 of the above algorithm
    if n.right is not None:
        return minValue(n.right)
 
    # Step 2 of the above algorithm
    p = n.parent
    while( p is not None):
        if n != p.right :
            break
        n = p 
        p = p.parent
    return p
 
# Given a non-empty binary search tree, return the 
# minimum data value found in that tree. Note that the
# entire tree doesn't need to be searched
def minValue(node):
    current = node
 
    # loop down to find the leftmost leaf
    while(current is not None):
        if current.left is None:
            
            break
        current = current.data
 
    return current
##############################################
#Check if a node is child of root node
#Use this to find if root is common ancestor 
class Node:
 
    # Constructor to create a new node
    def __init__(self, key):
        self.data = key 
        self.left = None
        self.right = None

        
def isChild(root, n):
    
    if root is None:
        return False
    
    if root == n:
        return True
    
    return isChild(root.left, n) or isChild(root.right, n)


##############################################

#first common ancestor of two nodes in a binary tree

def commonAncestor(root, p, q):
    if isChild(root.left, p) and isChild(root.left, q):
        return commonAncestor(root.left, p, q)
    if isChild(root.right, p) and isChild(root.right, q):
        return commonAncestor(root.right, p, q)
    return root
    
##############################################

# A recursive python program to find LCA of two nodes
# n1 and n2
 
# A Binary tree node
class Node:
 
    # Constructor to create a new node
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
 
# Function to find LCA of n1 and n2. The function assumes
# that both n1 and n2 are present in BST
def lca(root, n1, n2):
     
    # Base Case
    if root is None:
        return None
 
    # If both n1 and n2 are smaller than root, then LCA
    # lies in left
    if(root.data > n1 and root.data > n2):
        return lca(root.left, n1, n2)
 
    # If both n1 and n2 are greater than root, then LCA
    # lies in right 
    if(root.data < n1 and root.data < n2):
        return lca(root.right, n1, n2)
 
    return root
##############################################
# Python program to check binary tree is a subtree of 
# another tree
 
# A binary tree node
class Node:
 
    # Constructor to create a new node
    def __init__(self, data):
        self.data = data 
        self.left = None
        self.right = None
 
# A utility function to check whether trees with roots
# as root 1 and root2 are identical or not
def areIdentical(root1, root2):
     
    # Base Case
    if root1 is None and root2 is None:
        return True
    if root1 is None or root2 is None:
        return False
 
    # Check if the data of both roots is same and data of
    # left and right subtrees are also same
    return (root1.data == root2.data and
            areIdentical(root1.left , root2.left)and
            areIdentical(root1.right, root2.right)
            ) 
 
# This function returns True if S is a subtree of T,
# otherwise False
def isSubtree(T, S):
     
    # Base Case
    if S is None:
        return True
 
    if T is None:
        return True
 
    # Check the tree with root as current node
    if (areIdentical(T, S)):
        return True
 
    # IF the tree with root as current node doesn't match
    # then try left and right subtreee one by one
    return isSubtree(T.left, S) or isSubtree(T.right, S)
 


##############################################
#Bottom View of a Binary Tree
#Level order traversal with dictionary to store the horizontal distance from node 


# A binary tree node
class Node:
 
    # Constructor to create a new node
    def __init__(self, value):
        self.value = value 
        self.left = None
        self.right = None
        self.hd = None
 

def bottomView(root):
    
    if not root:
        return
    
    queue = [root]
    hd = 0
    root.hd = hd
    map_bw = {}
    
    while len(queue)>0:
        
        n = queue.pop(0)
        hd = n.hd
        
        #overwrite any previous entry for that horizontal distance
        #because it is now at an upper level and hidden by new level.
        map_bw[hd] = n.value
        
        if n.left:
            n.left.hd = hd - 1
            queue.append(n.left)
            
        if n.right:
            n.right.hd = hd + 1
            queue.append(n.right)
            
    for i in sorted(map_bw):  #sort the dictionary by key i.e. hd to print from left to right
        print (i,map_bw[i])
   

##############################################
#Top View of a Binary Tree
#Level order traversal with dictionary to store the horizontal distance from node 


# A binary tree node
class Node:
 
    # Constructor to create a new node
    def __init__(self, value):
        self.value = value 
        self.left = None
        self.right = None
        self.hd = None
 

def topView(root):
    
    if not root:
        return
    
    queue = [root]
    hd = 0
    root.hd = hd
    map_bw = {}
    
    while len(queue)>0:
        
        n = queue.pop(0)
        hd = n.hd
        
        #Do not over-write any previous entry for that horizontal distance
        #because it is entered at an upper level and should not be hidden by new level.
        if hd not in map_bw:
            map_bw[hd] = n.value
        
        if n.left:
            n.left.hd = hd - 1
            queue.append(n.left)
            
        if n.right:
            n.right.hd = hd + 1
            queue.append(n.right)
            
    for i in sorted(map_bw):  #sort the dictionary by key i.e. hd to print from left to right
        print (i,map_bw[i])
        


##############################################

# Find next node in same level for given node in a binary tree
class Node:
 
    # Constructor to create a new node
    def __init__(self, value):
        self.value = value 
        self.left = None
        self.right = None

def nextNode(root, v):
    
    queue = [root]
    level = 0
    level_q = [level] #to store level as well
    while len(queue)>0:
        
        n = queue.pop(0)
        level = level_q.pop(0) #take node's level
        
        if n.value == v:
            # If there are no more items in queue or given
            # node is the rightmost node of its level, 
            # then return None
            if len(queue) == 0 or level_q[0] != level:
                return "No next node"
            else:
                return queue[0].value
            
            if queue:
                return queue.pop(0).value
            else:
                return None
        
        if n.left:
            queue.append(n.left)
            level_q.append(level + 1 ) #add level too
        if n.right:
            queue.append(n.right)
            level_q.append(level + 1 )

##############################################
#Max Width of tree
#Pre order 

class Node:
 
    # Constructor to create a new node
    def __init__(self, value):
        self.value = value 
        self.left = None
        self.right = None

        
def preOrder(root, level, map_nodes):
    
    if root is None:
        return
    
    if level in map_nodes:
        map_nodes[level] += 1
    else:
        map_nodes[level] = 1
    
    if root.left:
        preOrder(root.left, level + 1,  map_nodes)
    if root.right:
        preOrder(root.right, level + 1,  map_nodes)
    
        
def maxwidth(root):
    
    map_n = {}
    width = 0
    
    preOrder(root,1,map_n)
    
    for i in map_n:
        
        width = max(width, map_n[i])
        
    return width
    
    
    
##############################################
# root-to-leaf paths of a binary tree
#Rrecursive

class Node:
 
    # Constructor to create a new node
    def __init__(self, value):
        self.value = value 
        self.left = None
        self.right = None


def rootToPath(root):
    p_arr = [0] * 10000 #   assume max length = 10000
    p_len = 0
    printPath(root, p_arr, 0 )

def printPath(node, path_arr, path_len ):
    
    if node is None:
        return
    
    path_arr[path_len] = node.value
    path_len += 1
    
    #if we reach leaf, print the path 
    if node.left is None and node.right is None:
        
#         print(path_arr) # we need to print the array till the path_len
        printPathArr(path_arr, path_len)
        print("")
    else:
        if node.left:
            printPath(node.left, path_arr, path_len)
        if node.right:
            printPath(node.right, path_arr, path_len)
    
def printPathArr(arr,len1):
    
    for i in range(len1):
        
        print (arr[i], end=" ")

r = Node(3)
a = Node(1)
b = Node(5)
c = Node(0)
d = Node(2)
e = Node(4)
f = Node(6)
g = Node(7)

r.left = a
r.right = b
a.left = c
a.right = d
b.left = e
b.right = f
c.left = g
""" TREE 1
     Construct the following tree
              3
            /   \
           1     5
         /  \   /  \
        0    2 4    6
       /
      7 
      
    """

rootToPath(r)
##############################################
#Invert a binary tree
#recursive
class Node:
 
    # Constructor to create a new node
    def __init__(self, value):
        self.value = value 
        self.left = None
        self.right = None

def invertTree(root):
    
    if not root:
        return None
    
    l_node = invertTree(root.left)
    r_node = invertTree(root.right)
    
    root.right = l_node
    root.left = r_node
#     if root.left and root.right:
#         root.left = r_node
#         root.right = l_node
        
#     if root.left:
#         root.right = l_node
#         root.left = None
#     if root.right:
#         root.left = r_node
#         root.right = None
        
    return root

##############################################
#Invert a binary tree
#iterative

def invertTreeIter(root):
    queue = [root]
    
    while len(queue)>0:
        
        n = queue.pop(0)
        
        temp = n.left
        n.left = n.right
        n.right = temp
        
        if n.left:
            queue.append(n.left)
            
        if n.right:
            queue.append(n.right)




##############################################
#Diameter of a binary tree.. maximum length between two nodes
#recursive
class Node:
 
    # Constructor to create a new node
    def __init__(self, value):
        self.value = value 
        self.left = None
        self.right = None

        
#level order traversal 
def diameterTree(root, diameter = None):
    
    if root is None:
        return 0
    
    if diameter is None:
        diameter = [1]
    
    
    leftHeight = diameterTree(root.left, diameter)
    rightHeight = diameterTree(root.right, diameter)
    
    max_diameter =  1 + leftHeight + rightHeight
    
#     diameter.insert(0,max(diameter[0], max_diameter)) 
    
    diameter[0] = max(diameter[0], max_diameter)
    
    print (diameter)
    return max(leftHeight, rightHeight) + 1
    
    
def maxDiameter(root):
    diameter = [0]
    diameterTree(root, diameter)
    return diameter

##############################################
#CHECK IF TREE IS SYMMETRIC

class Node:
 
    # Constructor to create a new node
    def __init__(self, value):
        self.value = value 
        self.left = None
        self.right = None

#check two nodes are symmetric        
def isSymmetricNodes(n1,n2):
    
    if n1 is None and n2 is None:
        return True
    
    if n1 and n2 and isSymmetricNodes(n1.left,n2.right) and isSymmetricNodes(n1.right,n2.left):
        return True
    
    return False

def isSymmetricTree(root):
    
    if root is None:
        return True
    
    return isSymmetricNodes(root.left, root.right)

##############################################


##############################################


##############################################


##############################################


##############################################


##############################################


##############################################


##############################################


##############################################


##############################################


##############################################


##############################################


