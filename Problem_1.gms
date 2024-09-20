Set s /1*2/;
Set i /1*8/;
Set j /1*5/;
Option seed =0
Parameter A(i,j)  the number of units of part j to produce a unit of product i;
Parameter D(s,i)  demand based on scenarios;

Scalar trail_number / 10 /;
Scalar p_s / 0.5 /;

Parameter b(j), l(i), q(i), r(j) ;

* Generate random values and assign to parameters
q(i) = UniformInt(900, 1000);
l(i) = UniformInt(600, 800);
b(j) = UniformInt(10, 15);
r(j) = UniformInt(5, 9);
D(s,i) = randBinomial(trail_number, p_s);
* Generate random values to vector A
A(i,j) = UniformInt(1, 10);

Variables
    x(j)       number of each type of parts to be preordered
    y(s, j)    number of parts left after in the inventory
    z(s, i)    number of products produced
    M the objective function ;  

Positive Variables x, y, z;

Equations
    profit 
    part_left(s,j)
    product_demand_req(s,i);


* Define the model
Model two_stage / all /;

File emp / '%emp.info%' /;
put emp '* problem %gams.i%';
$onput 
$offput

x.stage(j) = 1;
y.stage(s, j) = 2;
z.stage(s, i) = 2;

profit ..    M =e=  sum(j, b(j)*x(j)) + sum(s, 0.5*(sum(i,(l(i)-q(i))*z(s,i))-sum(j,r(j)*y(s,j))));
part_left(s, j) ..   y(s, j) =e= x(j) - sum(i, A(i,j) * z(s, i));
product_demand_req(s, i) ..      z(s, i) =l= D(s, i);

Solve two_stage using MIP minimizing M;
Display A, l, b, D, q, s, x.l, y.l, z.l, M.l;
Option Limrow=16;