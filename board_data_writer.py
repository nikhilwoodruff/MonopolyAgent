import pickle

fields = ['name', 'category', 'purchaseable']
int_fields = ['purchaseable']
properties = []

with open('monopoly_board.data', mode='rb') as f:
    board = pickle.loads(f.read())

response = ''
index = 0
while response != 'exit':
    current_property = {}
    for field in fields:
        response = input(field + ": ")
        current_property[field] = response
    for int_field in int_fields:
        current_property[int_field] = int(current_property[int_field])
    current_property['index'] = index
    index += 1
    if current_property['purchaseable'] == 1:
        current_property['buy_price'] = int(input('buy price: '))
        current_property['prices'] = []
        current_property['house_price'] = int(input('house price: '))
        while response != 'next' and response != 'exit':
            response = input('price with ' + str(len(current_property['prices'])) + ' houses:')
            if response != 'next' and response != 'exit':
                current_property['prices'].append(int(response))
    properties.append(current_property)

with open('monopoly_board.data', mode='wb') as f:
    f.write(pickle.dumps(properties))