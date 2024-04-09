from pyomo.environ import *

model = ConcreteModel()

# Sets
model.I = RangeSet(1, 3)  # Set of items
model.J = RangeSet(1, 2)  # Set of bins

# Parameters
model.c = Param(model.I, model.J, initialize={
    (1, 1): 5, (1, 2): 7,
    (2, 1): 3, (2, 2): 4,
    (3, 1): 2, (3, 2): 6
})  # Cost of assigning an item to a bin

model.b = Param(model.I, initialize={
    1: 2,
    2: 3,
    3: 1
})  # Maximum number of items that can be assigned to each item

# Variables
model.x = Var(model.I, model.J, domain=Binary)  # Assignment variables

# Objective function
model.obj = Objective(expr=sum(model.c[i, j] * model.x[i, j] for i in model.I for j in model.J), sense=minimize)

# Constraints
model.capacity = ConstraintList()
for j in model.J:
    model.capacity.add(sum(model.x[i, j] for i in model.I) <= 4)

# Solve the model
solver = SolverFactory('glpk')
results = solver.solve(model)

# Print the results
model.display()