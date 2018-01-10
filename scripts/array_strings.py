#Hints:
#1. When you need to compare two strings, use dictionary with key = element, value = count
#2. When you need to check with unique char, use set()


##############################################
# string compression  #
# Ex: (ABBCCCDDDDEEEEE = A1B2C3D4E5)

def str_compress(s):
    
    if len(s) == 0:
        return ""
    
    counter = 1
    last_char = s[0]
    final = ""
    
    for i in s[1:]:
        
        if last_char == i:
            counter += 1
        else:
            final = final + last_char + str(counter)
            last_char = i
            counter = 1
            
    final = final + last_char + str(counter)
    
    print(final)

s = 'ABBCCCDDDDEEEEE'
str_compress(s)

##############################################
#unique characters
#string 'abcde' --> true
#string 'abcdeee'  --> false

def uniq_char(s):
    chars = set()
    
    for i in s:
        if i in chars:
            return False
        else:
            chars.add(i)
            
    return True

s = 'abcdee'
uniq_char(s)

##############################################
#Anagram 
# 'abcd 1234' = '12abcd 34'
#Hint: when you need to compare, use dictionary with key = element, value = count
def anagram(s1,s2):
    
    s1 = s1.replace(' ','').lower()
    s2 = s2.replace(' ','').lower()
    
    
    if len(s1) == len(s2) == 0:
        return False
    
    if len(s1) != len(s2):
        return False
    
    
    dic = {}
    
    for i in s1:
        if i in dic:
            dic[i] += 1
        else:
            dic[i] = 1
            
    for i in s2:
        if i in dic:
            dic[i] -= 1
        else:
            return False
        
    for i in dic:
        if dic[i] > 0:
            return False
    
    return True

s1 = '  abcd 1234   '
s2 = '12abcd 34'

anagram(s1,s2)

##############################################
#Array Pair Sum
#Given an integer array, output all the unique pairs that sum up to a specific value x
#([1,3,2,2],4) --> (1,3), (2,2)

def pair_sum(arr,x):
    
    if len(arr)<2:
        return
    
    seen = set()
    output_pairs = []
    
    for i in arr:
        
        target = x - i
        
        if target in seen:
            output_pairs.append((min(i, target), max(i, target)))
            print(i, target)
        else:
            seen.add(i)
                
    return output_pairs

pair_sum([1,3,2,2,2,4,0],4)

##############################################
#missing element
#[1,2,3,4,5,6,7] [3,7,2,1,4,6]   --> 5 is the missing number
#Hint: when you need to compare, use dictionary with key = element, value = count

def missing_num(arr1,arr2):
    
    dic = {}
    
    for i in arr2:
        if i in dic:
            dic[i] += 1
        else:
            dic[i] = 1
            
    for i in arr1:
        if i in dic: # this is important to check if value is present
            if dic[i] == 0 :
                return i
            else:
                dic[i] -= 1
        else:
            return i

missing_num([1,2,3,4,5,6,7], [3,7,2,1,4,6])            
            
##############################################
#Largest Continuous Sum
# Try to store current max, final max, and continuous elements 
# in these type of questions

def largest_sum(arr):
    if len(arr) == 0:
        return 0
    
    curr_sum = max_sum = arr[0]
    
    continous_large_set = [arr[0]]
    final_set = [arr[0]]
    
    for i in arr[1:]:
        
        curr_sum += i
        continous_large_set.append(i)
        print('curr_sum ',curr_sum)
        print('max_sum ',max_sum)
        if curr_sum > max_sum:
            max_sum = curr_sum
            final_set = [e for e in continous_large_set] #way to create new list
            print('hi')
            
        print('1',final_set)
        if curr_sum < 0:
            curr_sum = 0
            continous_large_set.clear()
        print(continous_large_set)
        print(final_set)
        
    print("Final max_sum: ",max_sum)
    print("Final final_set: ",final_set)
        
    return final_set
    

largest_sum([-13, -100, 25, -1])

##############################################
#Reverse string (recursive Way)

def reverse_string(s):
    
    #base condition
    if len(s) <= 1:
        return s
    
    return s[-1:] + reverse_string(s[:-1])
    
reverse_string('123456')

#For iterative way, convert string to list and use below steps

##############################################
# Recursive python program to reverse an array
 
# Function to reverse A[] from start to end
def reverseList(A, start, end):
    if start >= end:
        return
    A[start], A[end] = A[end], A[start]
    reverseList(A, start+1, end-1)
    
    return A

 

##############################################

# Iterative python program to reverse an array
# Function to reverse A[] from start to end
def reverseList(A, start, end):
    while start < end:
        A[start], A[end] = A[end], A[start]
        start += 1
        end -= 1
        
    return A




##############################################
#remove duplicate characters in string

def remove_duplicate(s):
    if len(s) < 2:
        return
    
    final = []
    
    for i in s:
        if i not in final:
            final.append(i)
    
    return final

remove_duplicate('aabbccddee')

##############################################
def remove_duplicate2(s):
    lst=list(s)
    lst.sort()
    i = len(lst) - 1
    
    while i>0:
        if s[i] == lst[i-1]:
            lst.pop(i)
        i -= 1
    return ''.join(lst)

remove_duplicate2('aabbccddee')
##############################################
#Replace character in a string 
#replace spaces with %20
def replace_string(s):
    
    str=''
    for i in s:
        if i == ' ':
            str += '%20'
        else:
            str += i
            
    return str

replace_string('abc 123  xyz')
##############################################
#Check if one string s1 is rotation of other string s2
# s1 = '123456', s2 = '561234'

def string_rotation(s1,s2):
    
    if len(s1) == len(s2) and  len(s1) > 0:
        s1s1 = s1 + s1
        
        if s2 in s1s1:
            return True
    
    return False

string_rotation('123456','612345')

##############################################
#Check and print all subarrays with 0 sum exists or not

def zero_subarray2(arr):
    for i in range(len(arr)):
        curr_sum = 0

        subarray_index = []
        for j in range(i,len(arr)):
            curr_sum += arr[j]

            subarray_index.append(arr[j])
            if curr_sum == 0:

                print (subarray_index)
                
        
#Complexity : O(n^2) 
#Space O(n)

##############################################
#Rearrange the array with alternate high and low elements

def rearrange_array(arr):
    
    for i in range (1,len(arr),2):
        
        if arr[i-1] > arr[i]:
            
            temp = arr[i]
            arr[i] = arr[i-1]
            arr[i-1] = temp
            
        if arr[i+1] > arr[i]:
            temp = arr[i]
            arr[i] = arr[i+1]
            arr[i+1] = temp
            
    return arr

rearrange_array([1,2,3,4,5,6,7,8,9])

##############################################
#Sort binary array in linear time
#arr = [1,0,0,1,1,1,0,0]
def sort_binary_arr(arr):
    
    index_for_0 = 0
    #get number of 0s
    for i in range(len(arr)):
        if arr[i] < 1:
            index_for_0 += 1
    
    for i in range(index_for_0):
        arr[i] = 0

    for i in range(index_for_0,len(arr)):
        arr[i] = 1
        
    return arr
            
#Complexity : O(3*n) 
#Space O(1)

sort_binary_arr([1,0,0,1,1,1,0,0,0,0,0,0,1,1])

##############################################
#(Dutch national flag problem)
#Sort an array containing 0’s, 1’s and 2’s
#arr = [1,0,2,1,1,1,0,0,2]

def dutch_flag_problem(arr):
    
    count_0 = 0
    count_1 = 0
    
    for i in range(len(arr)):
        if arr[i] < 1:
            count_0 += 1
        elif arr[i] == 1:
            count_1 += 1
    
    index_2 = count_0 + count_1
    
    for i in range(count_0):
        arr[i] = 0

    for i in range(count_0,index_2):
        arr[i] = 1
        
    for i in range(index_2,len(arr)):
        arr[i] = 2
        
    return arr

dutch_nat_flag_problem([1,0,2,1,1,1,0,0,2,2])



##############################################
#(Dutch national flag problem)
#Sort an array containing 0’s, 1’s and 2’s
#arr = [1,0,2,1,1,1,0,0,2]

def swap_items(arr,i1,i2):
    temp = arr[i1]
    arr[i1] = arr[i2]
    arr[i2] = temp
            

def dutch_nat_flag_problem2(arr):
    
    mid = low = 0
    high = len(arr) -1
    
    while (mid <= high):
        
        if arr[mid] > 1:
            swap_items(arr,mid,high)
            high -= 1
        
        elif arr[mid] < 1:
            swap_items(arr,mid,low)
            low += 1
            mid += 1
        
        else:
            mid += 1
            
    return arr


#This algorithm runs in O(n) time because it only passes through the array once swapping the necessary elements in place.
dutch_nat_flag_problem2([1,0,2,1,1,1,0,0,2,2])

##############################################

# Python program to reverse a string with special characters
#Hint: Use start and end pointers (start < end pointer)
# Returns true if x is an aplhabatic character, false otherwise
def isAlphabet(x):
    return x.isalpha()
 
def reverse(string):
    LIST = toList(string)
 
    # Initialize left and right pointers
    r = len(LIST) - 1
    l = 0
 
    # Traverse LIST from both ends until
    # 'l' and 'r'
    while l < r:
 
        # Ignore special characters
        if not isAlphabet(LIST[l]):
            l += 1
        elif not isAlphabet(LIST[r]):
            r -= 1
 
        else:   # Both LIST[l] and LIST[r] are not special
            LIST[l], LIST[r] = swap(LIST[l], LIST[r])
            l += 1
            r -= 1
 
    return toString(LIST)
 
# Utility functions
def toList(string):
    List = []
    for i in string:
        List.append(i)
    return List
 
def toString(List):
    return ''.join(List)
 
def swap(a, b):
    return b, a
 
##############################################

#1. 
# Python program to check if a string is palindrome or not
#1) Find reverse of string
#2) Check if reverse and original are same or not.


# function which return reverse of a string
def reverse(s):
    return s[::-1]
 
def isPalindrome(s):
    # Calling reverse function
    rev = reverse(s)
 
    # Checking if both string are equal or not
    if (s == rev):
        return True
    return False


#2. Iterative Method
# function to check string is 
# palindrome or not 
def isPalindrome(str):
 
    # Run loop from 0 to len/2 
    for i in xrange(0, len(str)/2): 
        if str[i] != str[len(str)-i-1]:
            return False
    return True


#3. Recursive function for a palindrome

def isPalindrome(string) :
   if len(string) <= 1 :
      return True
   if string[0] == string[len(string) - 1] :
      return isPalindrome(string[1:len(string) - 1])
   else :
      return False


#4.
# A utility function to check if a string str is palindrome
def isPalindrome(string):
 
    # Start from leftmost and rightmost corners of str
    l = 0
    h = len(string) - 1
 
    # Keep comparing characters while they are same
    while h > l:
        l+=1
        h-=1
        if string[l-1] != string[h+1]:
            return False
 
    # If we reach here, then all characters were matching    
    return True
    
    


 
##############################################

#Print all palindromic partitions of a string
#Hint to get all sub strings --> 
# have two pointers start and end,  and one pointer x to move from end to start
# When pointer x comes reaches to start, move start = start + 1 and x = end again.

def isPalindrome(string):
 
    # Start from leftmost and rightmost corners of str
    l = 0
    h = len(string) - 1
 
    # Keep comparing characters while they are same
    while h > l:
        l+=1
        h-=1
        if string[l-1] != string[h+1]:
            return False
 
    # If we reach here, then all characters were matching    
    return True
    
def all_palindromes(string):

    left,right=0,len(string)
    j=right
    results=[]

    while left < right-1:
        temp = string[left:j] #Time complexity O(k)
        j-=1

        if isPalindrome(temp):
            results.append(temp)

        if j<left+2:
            left+=1
            j=right

    return results

print (all_palindromes("racecarenterelephantmalayalam"))

 
##############################################
# A Simple Python3 program to 
# count triplets with sum 
# smaller than a given value

#1 
def countTriplets(arr, n, sum):
 
    # Initialize result
    ans = 0
 
    # Fix the first element as A[i]
    for i in range(0, n-2):
     
        # Fix the second element as A[j]
        for j in range(i+1, n-1):
     
            # Now look for the third number
            for  k in range(j+1, n):
                if (arr[i] + arr[j] + arr[k] < sum):
                    ans += 1
    return ans
 
# Driver program
arr = [5, 1, 3, 4, 7]
n = len(arr)
sum = 12
print(countTriplets(arr, n, sum))

#2
# Time complexity of above solution is O(n3). 
# An Efficient Solution can count triplets in O(n2) by sorting the array first 

#Sort the array
#fix the first element  one by one and find the other two elements 
# j = i+1 , k = n-1
# move other two index : increment j if sum is less, decrement k is sum is more.

def find3Numbers(A, arr_size, sum):
 
    # Sort the elements 
    A.sort()
 
    # Now fix the first element 
    # one by one and find the
    # other two elements 
    for i in range(0, arr_size-2):
     
 
        # To find the other two elements,
        # start two index variables from
        # two corners of the array and
        # move them toward each other
         
        # index of the first element
        # in the remaining elements
        l = i + 1
         
        # index of the last element
        r = arr_size-1
        while (l < r):
         
            if( A[i] + A[l] + A[r] == sum):
                print("Triplet is", A[i], 
                     ',', A[l], ',', A[r]);
                return True
             
            elif (A[i] + A[l] + A[r] < sum):
                l += 1
            else: # A[i] + A[l] + A[r] > sum
                r -= 1
 
    # If we reach here, then
    # no triplet was found
    return False
    
 
##############################################
#Pythagorean Triplet in an array
#Given an array of integers, write a function that returns true 
#if there is a triplet (a, b, c) that satisfies a^2 + b^2 = c^2.


# Method 1 (Naive)
# O(n^3)

# Python program to check if there is Pythagorean
# triplet in given array
 
# Returns true if there is Pythagorean
# triplet in ar[0..n-1]
def isTriplet(ar, n):
    for i in range(n - 2):
        for k in range(j + 1, n):
            for j in range(i + 1, n - 1):
                # Calculate square of array elements
                x = ar[i]*ar[i]
                y = ar[j]*ar[j]
                z = ar[k]*ar[k] 
                if (x == y + z or y == x + z or z == x + y):
                    return 1
    
    # If we reach here, no triplet found
    return 0
  
# Method 2 (Use Sorting)
# O(n^2)

#1) Do square of every element in input array. This step takes O(n) time.
#2) Sort the squared array in increasing order. This step takes O(nLogn) time.
#3) To find a triplet (a, b, c) such that a = b + c, do following.
#	fix the first element  one by one and find the other two elements 
# 	j = i+1 , k = n-1
# 	move other two index : increment j if sum is less, decrement k is sum is more.

# Returns true if there is Pythagorean
# triplet in ar[0..n-1]
def isTriplet(ar, n):
    # Square all the elemennts
    for i in range(n):
        ar[i] = ar[i] * ar[i]
  
    # sort array elements
    ar.sort()
  
    # fix one element
    # and find other two
    # i goes from n - 1 to 2
    for i in range(n-1, 1, -1):
        # start two index variables from 
        # two corners of the array and 
        # move them toward each other
        j = 0
        k = i - 1
        while (j < k):
            # A triplet found
            if (ar[j] + ar[k] == ar[i]):
                return True
            else:
                if (ar[j] + ar[k] < ar[i]):
                    j = j + 1
                else:
                    k = k - 1
    # If we reach here, then no triplet found
    return False
    


##############################################
# Length of the largest subarray with contiguous elements
#Method 1(Brute Force)
 class Solution:
    def longestConsecutive(self, nums):
        longest_streak = 0

        for num in nums:
            current_num = num
            current_streak = 1

            while current_num + 1 in nums:
                current_num += 1
                current_streak += 1

            longest_streak = max(longest_streak, current_streak)

        return longest_streak
        
#Method 2:  Sorting
# Time complexity O(NLogN)
class Solution:
    def longestConsecutive(self, nums):
        if not nums:
            return 0

        nums.sort()

        longest_streak = 1
        current_streak = 1

        for i in range(1, len(nums)):
            if nums[i] != nums[i-1]:
                if nums[i] == nums[i-1]+1:
                    current_streak += 1
                else:
                    longest_streak = max(longest_streak, current_streak)
                    current_streak = 1
        
        return max(longest_streak, current_streak)


#Method 3 (O(n)): HashSet and Intelligent Sequence Building
class Solution:
    def longestConsecutive(self, nums):
        longest_streak = 0
        num_set = set(nums)

        for num in num_set:
            if num - 1 not in num_set:
                current_num = num
                current_streak = 1

                while current_num + 1 in num_set:
                    current_num += 1
                    current_streak += 1

                longest_streak = max(longest_streak, current_streak)

        return longest_streak
        
##############################################
#Find the smallest positive integer value that cannot 
# be represented as sum of any subset of a given array

# Returns the smallest number 
# that cannot be represented as sum
# of subset of elements from set
# represented by sorted array arr[0..n-1]
def findSmallest(arr, n):
 
    res = 1 #Initialize result
 
    # Traverse the array and increment
    # 'res' if arr[i] is smaller than
    # or equal to 'res'.
    for i in range (0, n ):
        if arr[i] <= res:
            res = res + arr[i]
        else:
            break
    return res


 
##############################################
#Stock Buy Sell to Maximize Profit
# (1) Iterate through each number in the list.
# (2) At the ith index, get the i+1 index price and check if it is larger than the ith index price.
# (3) If so, set buy_price = i and sell_price = i+1. Then calculate the profit: sell_price - buy_price.
# (4) If a stock price is found that is cheaper than the current buy_price, set this to be the new buying price and continue from step 2.
# (5) Otherwise, continue changing only the sell_price and keep buy_price set.



def StockPicker(arr): 
  
  max_profit = -1
  buy_price = 0
  sell_price = 0
  
  # this allows our loop to keep iterating the buying
  # price until a cheap stock price is found
  change_buy_index = True
  
  # loop through list of stock prices once
  for i in range(0, len(arr)-1):
    
    # selling price is the next element in list
    sell_price = arr[i+1]
    
    # if we have not found a suitable cheap buying price yet
    # we set the buying price equal to the current element
    if change_buy_index: 
      buy_price = arr[i]
    
    # if the selling price is less than the buying price
    # we know we cannot make a profit so we continue to the 
    # next element in the list which will be the new buying price
    if sell_price < buy_price:
      change_buy_index = True 
      continue
    
    # if the selling price is greater than the buying price
    # we check to see if these two indices give us a better 
    # profit then what we currently have
    else:
      temp_profit = sell_price - buy_price
      if temp_profit > max_profit:
        max_profit = temp_profit
      change_buy_index = False
      
  return max_profit

print StockPicker([44, 30, 24, 32, 35, 30, 40, 38, 15])

##############################################

 
##############################################




