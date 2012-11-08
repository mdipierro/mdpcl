# pyopencl interface with python->c99 compiler
# created by Massimo Di Pierro
# license: 2-clause BSD
# requires meta and pyopencl

import logging
import pyopencl
import sys
import ast
import numpy
try:
    from meta.decompiler import decompile_func
except ImportError:
    logging.error('need to install meta')

class Device(object):
    flags = pyopencl.mem_flags
    def __init__(self):
        self.ctx = pyopencl.create_some_context()
        self.queue = pyopencl.CommandQueue(self.ctx)
        self.define=Compiler()
    def buffer(self,source=None,size=0,mode=pyopencl.mem_flags.READ_WRITE):
        if source is not None: mode = mode | pyopencl.mem_flags.COPY_HOST_PTR
        buffer = pyopencl.Buffer(self.ctx, mode, size=size, hostbuf=source)
        return buffer
    def retrieve(self,buffer,shape=None,dtype=numpy.float32):
        output = numpy.zeros(shape or buffer.size/4,dtype=dtype)
        pyopencl.enqueue_copy(self.queue, output, buffer)
        return output
    def compile(self,kernel):
        return pyopencl.Program(self.ctx,kernel).build()

class C99Handler(object):
    sep = ' '*4    
    @staticmethod
    def make_type(name):
        newname = stars = ''
        if name.startswith('global:'):
            newname, name = newname+'__global ' , name[7:]
        elif name.startswith('local:'):
            newname, name = newname+'__local ' , name[6:]
        if name.startswith('const:'):
            newname, name = newname+'const ' , name[6:]
        while name.startswith('ptr_'):
            stars, name = stars+'*' , name[4:]
        return newname+name+stars
    def is_list(self,items,pad):
        t_items = [self.t(item,pad) for item in items]
        return pad+('\n'+pad).join(t for t in t_items if t.strip())
    def is_FunctionDef(self,item,types,pad):
        args = ', '.join('%s %s' % (self.make_type(types[a.id]),a.id) 
                         for a in item.args.args)
        return 'void %s(%s) {\n/*@VARS*/\n%s\n}' % (
            item.name, args, self.t(item.body,pad+self.sep))
    def is_Name(self,item,pad):
        if not item.id in self.constants:
            return item.id
        return self.represent(self.constants[item.id])
    def is_For(self,item,pad):
        if not item.iter.func.id == 'range': raise NotImplementedError
        args = item.iter.args
        if len(args)==1:
            start, stop, incr = 0, self.t(args[0],pad), 1
        elif len(args)==2:
            start, stop, incr = self.t(args[0],pad), self.t(args[1],pad), 1
        elif len(args)==3:
            start, stop, incr = self.t(args[0],pad), self.t(args[1],pad), self.t(args[2],pad)
        else: raise NotImplementedError
        if isinstance(item.target,ast.Name) and not item.target.id in self.symbols:
            self.symbols[item.target.id] = 'long'
        return 'for (%(n)s=%(a)s; %(n)s<%(b)s; %(n)s+=%(c)s) {\n%(d)s\n%(p)s}' % dict(
            n=self.t(item.target), a=start, b=stop, c=incr, 
            d=self.t(item.body,pad+self.sep),p=pad)
    def is_If(self,item,pad):        
        code = 'if (%(c)s) {\n%(b)s\n%(p)s}' % dict(
            c=self.t(item.test), b=self.t(item.body,pad+self.sep), p=pad)
        if item.orelse:
            code += ' else {\n%(e)s\n%(p)s}' % dict(
                e=self.t(item.orelse,pad+self.sep), p=pad)
        return code
    def is_Compare(self,item,pad): 
        return '(%s %s %s)' % (self.t(item.left),
                               self.t(item.ops[0]),
                               self.t(item.comparators[0]))
    def is_BoolOp(self,item,pad):
        return '(%s %s %s)' % (self.t(item.values[0]),
                               self.t(item.op),
                               self.t(item.values[1]))
    def is_Not(self,item,pad): return '!'
    def is_Or(self,item,pad): return '||'
    def is_And(self,item,pad): return '&&'
    def is_Eq(self,item,pad): return '=='
    def is_Gt(self,item,pad): return '>'
    def is_Lt(self,item,pad): return '<'
    def is_GtE(self,item,pad): return '>='
    def is_LtE(self,item,pad): return '<='
    def is_IsNot(self,item,pad): return '!='
    def is_NotEq(self,item,pad): return '!='        
    def is_Assign(self,item,pad):
        left, right = item.targets[0], item.value
        if isinstance(left,ast.Name) and not left.id in self.symbols:
            if isinstance(right,ast.Call) and right.func.id.startswith('new_'):
                self.symbols[left.id] = right.func.id[4:]                
                right = right.args[0] if right.args else None
            else:
                raise RuntimeError('unkown C-type %s' % left.id)
        return '%s = %s;' % (self.t(item.targets[0]),self.t(right)) if right else ''
    def is_Call(self,item,pad):
        return '%s(%s)' % (self.t(item.func),','.join(self.t(a) for a in item.args))
    def is_Num(self,item,pad):
        return self.represent(item.n)
    def is_Str(self,item,pad):
        return self.represent(item.s)
    def is_UnaryOp(self,item,pad):
        return '(%s %s)' % (self.t(item.op),self.t(item.operand))
    def is_BinOp(self,item,pad):
        return '(%s %s %s)' % (self.t(item.left),self.t(item.op),self.t(item.right))
    def is_USub(self,item,pad): return '-'
    def is_Add(self,item,pad): return '+'
    def is_Sub(self,item,pad): return '-'
    def is_Mult(self,item,pad): return '*'
    def is_Div(self,item,pad): return '/'
    def is_Subscript(self,item,pad):
        return '%s[%s]' % (self.t(item.value),self.t(item.slice.value))
    def is_Attribute(self,item,pad):
        return '%s.%s' % (self.t(item.value),item.attr)
    def is_Return(self,item,pad):
        self.rettype = self.symbols.get(item.value.id,None)
        return 'return %s;' % self.t(item.value) if self.rettype else ''
    def is_Expr(self,item,pad):
        return self.t(item.value)+';'
    def represent(self,item):
        if item is None:
            return 'null'
        elif isinstance(item,str):
            return '"%s"' % item.replace('"','\\"')
        else:
            return str(item)
    def __init__(self):
        self.constants = {}
        self.actions = { list: self.is_list }
        for key in dir(self):
            if key.startswith('is_') and key != 'is_list':
                self.actions[getattr(ast,key[3:])] = getattr(self,key)
    def compile(self,item,types,prefix=''):
        self.rettype = None # the return type of the function None is void
        self.symbols = {}
        code = self.is_FunctionDef(item,types,'')
        vars = ''.join('%s%s %s;\n' % (self.sep,self.make_type(v),k) 
                       for k,v in self.symbols.items())
        code = code.replace('/*@VARS*/',vars)
        if self.rettype is not None:
            code = self.rettype+code[4:]
        return prefix+code.replace('\n\n','\n')
    def t(self,item,pad=''):
        return self.actions[type(item)](item,pad)


class Compiler(object):
    def __init__(self,handler=C99Handler()):
        self.functions = {}
        self.handler = handler
    def __call__(self,prefix='',name=None,**types):
        if prefix == 'kernel': prefix='__kernel '
        def wrap(func,types=types,name=name,prefix=prefix):
            if name is None: name = func.__name__
            decompiled = decompile_func(func)
            self.functions[name] = dict(func=func, prefix=prefix, ast=decompiled,types=types)
            return func
        return wrap
    def getcode(self,headers=True,constants=None):
        if constants: self.handler.constants.update(constants)
        defs, funcs = [], []
        for name, info in self.functions.iteritems():
            code = self.handler.compile(info['ast'],info['types'],info['prefix']) 
            info['code'] = code
            if headers: defs.append(code.split(' {',1)[0]+';')
            funcs.append(code)
        return '\n\n'.join(defs+funcs)


def test():
    c99 = Compiler()

    @c99('kernel',a='global:ptr_float',b='global:ptr_float')
    def f(a,b):
        w = new_int(0)
        c = new_int()
        x = new_ptr_float(malloc(1000))
        for k in range(0,3,1):
            if k>0 or k<=1 or k is not 1 or k != 1:
                c = g(1)
            elif not k==2:
                c = -a + a+b*b
            else:
                c[1+k] = a.y(a)
        printf('hello "%s"',world)
        delete(x)
        return c
    return c99.getcode(headers=False)

if __name__ == '__main__':
    print test()
