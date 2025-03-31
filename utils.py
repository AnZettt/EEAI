def save(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

def load(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def clean(doc):
    import string

    doc = doc.replace('\n', '')
    table = str.maketrans('','',string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word.lower() for word in tokens]
    return tokens