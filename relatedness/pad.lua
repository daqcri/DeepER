require 'nn'
require 'nngraph'



x = nn.Identity()()
xs = nn.SplitTable(1)(x)

xm = nn.CAddTable()(xs)

model = nn.gModule({x}, {xm})

input = torch.Tensor(3,10):fill(1)

output = model:forward(input)/3


print(input)
print(output)
