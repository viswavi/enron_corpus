# MySQL commands to execute before running Python script
#
# mysql> CREATE DATABASE enron;
# mysql> USE enron;
# mysql> CREATE TABLE people (fileID VARCHAR(20) NOT NULL, sender VARCHAR(100), recipient VARCHAR(100), number_of_recipients INTEGER);
# mysql> CREATE TABLE messages (fileID VARCHAR(20) NOT NULL, sender VARCHAR(100), recipients TEXT, number_of_recipients INTEGER, subject TEXT, sent_time DATETIME, content MEDIUMTEXT, genre VARCHAR(20), information VARCHAR(20), topics VARCHAR(20), emotion VARCHAR(20));
#  ^here I am assuming that the email body is at most 16MB, because I'm defining
#   it as a MEDIUMTEXT type in SQL
import mysql.connector
import email
import os
import re
from dateutil import parser

# we are using two SQL tables (people and messages, defined above)
# to do our analysis here

def content_to_message_fields(content):
    msg = email.message_from_string(content)
    sender = msg['From']
    # split the recipients string on whitespace or comma
    recips = [msg['To']]
    if 'Cc' in msg:
        recips.append(msg['Cc'])
    if 'Bcc' in msg:
        recips.append(msg['Bcc'])
    try:
        all_recipients = ','.join(recips)
        split_recipients = re.split(', |\*|\n|\t|\r', all_recipients)
    except:
        return
    # and filter empty strings from the result
    recipients = [x for x in split_recipients if x]
    message_id = msg['Message-ID']
    subject = msg['Subject']
    body = msg.get_payload()
    # parse date of email and format in SQL-friendly datetime format
    dt = parser.parse(msg['Date']).strftime('%Y-%m-%d %H:%M:%S')
    return (sender, recipients, subject, dt, body)

#turn the text of categories into a list of fixed size
#where the i^th element is the label id of that category
def parse_categories(content):
    parse_label = lambda x: (int(x.split(',')[0]), int(x.split(',')[1]))
    cats = [parse_label(x) for x in content.split("\n") if x.split()]
    x = [[], [], [], []]
    for (label_index, value) in cats:
        x[label_index-1].append(str(value))
    x = [",".join(values) if values else None for values in x]
    return x

def write_to_messages_table(cursor, file_id, sender, recipients, subject, dt, body, genre, info, topics, emotion):
    add_message = ("INSERT INTO messages (fileID, sender, recipients, number_of_recipients, subject, " +
               "sent_time, content, genre, information, topics, emotion) VALUES " +
               "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)")
    num_recipients = len(recipients)
    recip_str = ','.join(recipients)
    fields = (file_id, sender, recip_str, num_recipients, subject, dt, body, genre, info, topics, emotion)
    cursor.execute(add_message, fields)

def write_to_people_table(cursor, file_id, sender, recipients):
    add_message = ("INSERT INTO people (fileID, sender, recipient, number_of_recipients)" +
               " VALUES (%s, %s, %s, %s)")
    for recipient in recipients:
        fields = (file_id, sender, recipient, len(recipients))
        cursor.execute(add_message, fields)

cnx = mysql.connector.connect(user='root',database='enron',password='Cheerio627')
cursor = cnx.cursor()

def populate_tables():
    directories = ['enron_with_categories/{}/'.format(i) for i in list(range(1,9))]
    for d in directories:
        files = os.listdir(d)
        file_ids = set([x.split('.')[0] for x in files])
        for file_id in file_ids:
            text = open(d + file_id + ".txt").read()
            parsed = content_to_message_fields(text)
            if(not parsed):
                continue
            else:
                (sender, recipients, subject, dt, body) = content_to_message_fields(text)
            cats_text = open(d + file_id + ".cats").read()
            [genre, info, topics, emotion] = parse_categories(cats_text)
            write_to_messages_table(cursor, file_id, sender, recipients, subject,
            dt, body, genre, info, topics, emotion)
            write_to_people_table(cursor, file_id, sender, recipients)
    cnx.commit()

def top_recipients_of_DMs():
    query = ("select recipient, COUNT(recipient) as c from " +
            "(select * from people where number_of_recipients = 1) as DM " +
            "GROUP BY recipient ORDER BY c DESC LIMIT 3;")
    cursor.execute(query)
    print "\nrecipients receiving the largest number of direct mails:\n"
    for (name, count) in cursor:
        print "{} received {} direct mails".format(name,count)

def top_senders_of_broadcasts():
    query = ("select sender, COUNT(sender) as c from " +
            "(select * from messages where number_of_recipients > 1) as broadcast " +
            "GROUP BY sender ORDER BY c DESC LIMIT 3;")
    cursor.execute(query)
    print "\nsenders sending the largest number of broadcast mails:\n"
    for (name, count) in cursor:
        print "{} sent {} broadcast mails".format(name,count)

def fastest_responses():
    query = ["select b.fileID as response_id, a.fileID as original_id,",
             "b.subject as subject,",
             "b.sender AS response_sender, a.sender AS original_sender,",
             "TIMEDIFF(b.sent_time, a.sent_time) AS response_time from",
             "(messages a INNER JOIN messages b ON",
             "(FIND_IN_SET(b.sender, a.recipients) > 0)",
             "AND INSTR(b.subject, a.subject) > 0",
             "AND a.sent_time < b.sent_time)",
             "WHERE b.sender != a.sender",
             "ORDER BY response_time ASC",
             "LIMIT 5;"]
    cursor.execute(" ".join(query))
    print "\nfastest response times:\n"
    for (response_id, original_id, subject, responder, originator, response_time) in cursor:
        print "{} {} {} {} {} {}".format(response_id, original_id, subject, responder,
        originator, response_time)

if __name__ == "__main__":
    populate_tables()
    top_recipients_of_DMs()
    top_senders_of_broadcasts()
    fastest_responses()
