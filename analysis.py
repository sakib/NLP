#load text files containing data and compare them
from difflib import SequenceMatcher

all_books = []
for i in range(1,8):
    with open('data/hp/hp{}.txt'.format(i),'r') as book:
        #print (book)
        all_books = all_books + (book.readlines())

#gen = open('gen.txt', 'r')
#gen_s = ''.join(gen.readlines())

gen_s = '"I want to know what he\'s about to do." \
    Harry felt a seat at the back of Kreacher\'s neck she had chosen to speak. \
    "Do you know what I think?" said Harry, staring at the pair of them. \
    "We won\'t be sure that you were aware of everything they have seen in the Pensieve." \
    The crowd below was sweaty trunks, some looking pearly-white and staring. \
    Harry stared at him in an instant he was pointing at the pair of them, \
    the people in the crowd cheered. They had been excited by the afterpeering theor in the...'

all_books_s = ''.join(all_books)

sm = SequenceMatcher(None, gen_s, all_books_s)
print (sm.ratio())

for block in sm.get_matching_blocks():
    #print ('Generated text (%d) and HP (%d) match for % elements' %block)
    print (block)
    #print ('{} \n and \n {} match \n'.format(gen_s[block[0]:block[0] + block[2]], \
            #all_books[block[1]:block[1] + block[2]]))
