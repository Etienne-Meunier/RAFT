from ipdb import set_trace


def salut(x) :
    print(x)
    set_trace()
    print('test')


if __name__=='__main__' :
    salut(5)
