#!/usr/bin/env python
# coding: utf-8

# In[1]:

# Importing required NLP Python Libraries for Identification with High Accuracy and Low False Identification Rate

import numpy
import csv

import spacy  # A dedicated library for NLP Purposes for practical applications
nlp = spacy.load('en_core_web_sm')

import geograpy  # A Library specifically designed to identify locations based on NLTK
import nltk      # A dedicated library for NLP Purposes 
nltk.downloader.download('averaged_perceptron_tagger')

from time import perf_counter  # Measuring Performance (Time) a.k.a Cost of Implementation of Algo used


# In[2]:

# String with the given text 

text = """
Dynasty Newsletter #30 (6.7.20)
Welcome Our New Section Mate - Dan Wenick!  + Section Spotlights Ft. Kelleigh, Justin, Abhay
It is my absolute pleasure to let y'all know that we have a NEW SECTION MATE! And so the Dynasty keeps growing... :D

As you know, some students defer between RC/EC year and different semesters, once they come back they're reassigned a section for graduating/for LIFE! Please join me in welcoming, Dan Wernick to the D-NASTY section! 




Dan's intro to y'all below - feel free to shoot him a welcome note at dwernick@mba2021.hbs.edu!  
+ SECTION and HBS NEWS:
VIRTUAL SOCIAL HOUR kickoff this Tuesday (6/9) @ 5:45 - 6:45 PM EST!!! Join us as we catch up with each other over Zoom
Also, friendly reminder anyone and everyone is encouraged to plan social events (local or virtual) for the section
Nga has planned a laDies Paint & Sip with Cata next Sat (6/13) - all laDies welcome!   
HAPPY EARLY BDAY to Andrew V, (6/8), Aditi (6/10), Bill (6/13), Ruth (6/13)
Wednesday (6/10) @  9AM EST - Jan and Jana will host the first summer update on Fall Planning decisions. (zoom link in your section calendar email)
Lastly, I wanted to give a special shout out and thanks to Natalie and Steph for their incredibly powerful and vulnerable emails. I am sorry the world is in such chaos and that there is so much ongoing injustice that directly impacts y'all.  We are here for you, now and always <3

Dynasty - apologies for the multiple emails. One final announcement I did not want included in the broader newsletter distro. I just found out Kwei has been recently hospitalized with Dengue fever in Indo. He's doing fine and recovering now but was hospitalized for 5 days. To the extent you feel comfortable, any notes or thoughts to him wishing him a well and speedy recovery would be great. 

PS. Lookout for communication from Tarun regarding a continuing conversation on the subject we hope to have with the section and professors. I'm extremely proud that Dynasty really shined in our CC throughouth the year by engaging and respecting each other in the tough conversations. Let's continue to be there for each other.  

Dan Wernick   (dwernick@mba2021.hbs.edu)

Hi Section (D)ynasty! I hope you all had a great RC year and are settling into your summer situations as comfortably as possible. By way of quick background, I have the great privilege of joining you all for EC Year. I originally entered HBS as part of the Class of 2019 (Section D), but left after RC year to work on a startup in the travel space called PlacePass, where I have spent the past two years. Given the current state of travel, I have decided that now is the best time to return to HBS to complete EC Year, and I am absolutely pumped to be joining Section (D)ynasty! While I have been living in Boston for the past 7+ years, I am now splitting time between Boston and Longmeadow, MA where I grew up – one perk of being able to work remotely! For those currently in the Boston area, feel free to shoot me an email if you’d like to meet up (socially distanced of course). For those who aren’t, I hope to have the opportunity to meet you in person at some point over the course of this year (ideally this fall!).


KELLEIGH   (looking fabulous as always!)   

First of all I want to thank Natalie and Stephanie for their extremely thoughtful emails this week. I am working to educate myself on how I can be more supportive of the Black community and fight racial injustice. I recognize my privilege. I just wanted to add that note to use this platform to not be silent. ?
Words to live by:  “Work hard and be nice to people” - Anthony Burrill  
Where will you be spending summer? Mostly between Rhode Island and Connecticut, but hoping to come to Boston for some weekends!
What are your summer plans? I am lucky enough to have two internships this summer: first, I am working at a clothing company called Faherty Brand essentially in a Chief-of-Staff type role, and next I will be working in corporate strategy for Barry’s Bootcamp.
What's your quarantine update? I have officially realized that I am someone who can not relax. Since being in quarantine I have tried to pick up the ukulele, teach myself Adobe Premiere to make videos, help an HKS alum write a book, be an Upkey mentor for students who lost their internships… and I’m in the middle of Whole30. It’s self-inflicted busy-ness but it makes me excited to be alive! And I’m so, so thankful to be able to have the resources and support to be able to spend quarantine the way I am.
Best concert you've been to: Paul McCartney at Madison Square Garden when I was growing up.
Best memory of RC year: Summiting Mt. Kilimanjaro.
Your self proclaimed section superlative: Most likely to be on a stationary bike during Zoom class?

JUSTIN   (with his beautiful fiancee Victoria!)

Words to live by:  
No dress rehearsal, this is our life (lyric from “Ahead by a Century” by the Tragically Hip, an iconic Canadian rock band)  
Where will you be spending summer? 
Boston
What are your summer plans? 
Working at a Real Estate PE firm based in Boston, will be stationed at the SFP office
Looking forward to an annual golf weekend with my college buddies in August
What's your quarantine update? 
My fiancé, Victoria, and I have been dog-sitting her sisters Cavapoo, Layla, which has been a lot of fun…Vic’s putting the pressure on to get one of our own now. Stay tuned..
Annoying Vic with the same 5 songs I know on the guitar
Going for a run / bike ride along the Charles
Best concert you've been to: 
Bruce Springsteen on Broadway
Neil Young at Massey Hall
Tom Petty at MSG
Best memory of RC year: 
Section D coming out to support Orr and me embarrass ourselves at the HBS-Yale hockey game
Flag Day (shout out Korn) was a great reminder of all the unique perspectives and backgrounds in our section  
Being seatmates with Anoothi

ABHAY  pic of the wildlife in Sugar Land, TX... what I see on my daily walks :D 

Words to Live By: "When you’re serious about something, you build your life around it, instead of just fitting it into your life.”
Where will you be spending summer? Sugar Land, TX —the land of milk and honey
What are your summer plans? Working for a mindfulness/self-inquiry nonprofit called the TAT Foundation, helping them craft strategy for the next few years. 
What's your quarantine update? Taking daily walks! So so so necessary, and something I hope to continue with in my everyday life from now on. Adds a lot of peace to my day. Also been watching embarrassing amounts of Brooklyn 99 ??.
Best concert you've been to:  Kendrick Lamar, right after good kid m.a.a.d. city dropped—to this day, nothing has topped that
Best memory of RC year: In class, I think it was when Professor Mayo was super vulnerable with us in class—my only thought was like “this is what a great man looks like.” Made me want to be a better person. Outside of class, it was definitely EKTA, oh sneppp section D crushed it, and I felt honored to share the stage with these BALLERS.
Your self-proclaimed section superlative: Most likely to dance till he drops.
DYNASTY RESOURCES
Dynasty Summer City Tracker and Internships: Google Doc
Plus checkout the Outlook groups of regions for summer 
NYC; dynastynyc2020@groups.hbs.edu
Boston: dynastyboston2020@groups.hbs.edu
Chicago: dynastychicago2020@groups.hbs.edu
San Francisco: dynastysf2020@groups.hbs.edu
Asia: dynastyasia2020@groups.hbs.edu

DYNASTY's BOOK RECOMMENDATIONS (Professors + Section Mates)
Check out THIS summer reading hit list for the compiled list of book recommendations from our first and second-semester professors. Please feel free to add your own recs! 
DYNASTY's RECIPE COOKBOOK 
Drop in your favorite recipes here... we'll keep adding Christina's cooking for sure! 

CAREER RESOURCES (DYNASTY DROPBOX)
Dynasty's Interview Prep Dropbox is LIVE! Reach out to Aditi if you have files to upload. Dropbox login information (for file downloads): 
email: section.d.2021@gmail.com
Pword: Dynasty2019!
WHATSAAAAPP?!
Section Whatsapp (for social talk and comedic relief): https://chat.whatsapp.com/H9lOT3V0Zew8spRaNmHGiH
Dynasty laDies Whatsapp: https://chat.whatsapp.com/EVmh9QjMpLUAmy7uVsE7MP
Dynasty Dudes (laDs): https://chat.whatsapp.com/HKEBQTFum3D6GxmeI234Rw
D-stress Whatsapp: https://chat.whatsapp.com/D0lUAjLfn0JHc8GxJtNdPN
Email listservs
Student Section listserv (students only): mba2021d@listserv.hbs.edu
Section listserv (students + partners): sectiond2021@groups.hbs.edu
laDies listserv: dynastyladies@groups.hbs.edu
laDs listserv: DynastyLaDs@groups.hbs.edu
Dynasty International Students: SectionDInternationalStudents@groups.hbs.edu
Dynasty Leadership: SectionDLeadership@groups.hbs.edu

Birthdays & Partner Emails
Section birthdays and partner emails: https://docs.google.com/spreadsheets/d/1gfW3Ref3TrXfcHz0zZkez0gywFBZ70by68G2UKeSqkg/edit#gid=0
view this email in your browser


This email was sent to mba2021d@listserv.hbs.edu
why did I get this?    unsubscribe from this list    update subscription preferences
HBS , 2 Peabody Terr , 712 , Cambridge, MA 02138 · USA

Email Marketing Powered by Mailchimp

"""


# In[3]:


t_start = perf_counter()

gpe = []

doc = nlp(text)
places = geograpy.get_place_context(text=text)

for ent in doc.ents:
    if (ent.label_ == 'GPE'):
        gpe.append(ent.text)
    elif (ent.label_ == 'LOC'):
        gpe.append(ent.text)
    
for ent in places.cities:
    if (ent not in gpe):
        gpe.append(ent)
    
for ent in places.other:
    if (ent not in gpe):
        gpe.append(ent)
    
for ent in places.regions:
    if (ent not in gpe):
        gpe.append(ent)


# In[4]:

# To lower the False Identification Rate I have gone from a Database Cross-Validation for the given task.
# The sources for the cities.csv file is given below. The sources are well certified and known for their authenticity to be used for validation purposes.
# https://simplemaps.com/data/us-cities
# https://simplemaps.com/data/world-cities
# The above 2 sources were combined into a single csv file for ease of computation purposes.

data = []
with open("cities.csv", encoding = "utf-8") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        data.append(row)
        
arr = numpy.array(data)

print(arr)

final = []

for text in gpe:
    if (text in arr and (text not in final)):
        final.append(text)
        
print(final) ## Prints Final List of Cities, States or any other regions denoting any location
        
t_stop = perf_counter()  # Stops Time counter at end of Identification Process


# In[5]:

# The section below was specifically designed for the given doc to test 3 most inportant parameters : Accuracy, Cost (time) and False Identification nos.

answers = ['Indo', 'Boston','Longmeadow', 'MA', 'Rhode Island', 'Connecticut', 'Sugar Land', 'TX', 'Brooklyn', 'NYC', 'Chicago', 'San Francisco', 'Cambridge']

count = len(answers)
correct = 0
false = 0
for text in final:
    if text in answers:
        correct += 1
    else:
        false += 1
        
accuracy = correct/count *100
time = t_stop - t_start

print("Correct:",correct)
print("Ideal:",count)
print("Accuracy:",accuracy)
print("False Identification:", false)
print("Time Elapsed during Identification: ",time,"sec")


# In[ ]:




