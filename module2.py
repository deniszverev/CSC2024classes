#Write a program that calculates the total amount of a meal purchased at a restaurant. 
#
#The program should ask the user to enter the charge for the food. 
#
#Then calculate the amounts with an 18 percent tip and 7 percent sales tax. 
#
#Display each of these amounts and the total price.

total = 0
receipt = [

def switch (food, x):
    if food == "exit": 
        print('RECEIPT:\n', receipt
        exit()
    else :
        receipt.append([food, '@', '{:.2f}'.format(x)])


def main():
    while True:
        print('Enter a food item or the word exit:')
        food = input()
        print('Enter the price:')
        x = float(input())
        
        switch(food, x)

if __name__ == "__main__":
    main()