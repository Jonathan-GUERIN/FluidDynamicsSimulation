function y=f1_phys(rho)

y=a*rho

endfunction

function y=f2_phys(rho)

y=b*rho

endfunction

function y=a1_phys(rho)

y=a

endfunction

function y=a2_phys(rho)

y=b

endfunction

function y=q1_num(rhol,rhor)

if ischema==1
y=abs(a)
end

if ischema==2
y=a*dt
end

endfunction

function y=q2_num(rhol,rhor)

if ischema==1
y=abs(b)
end

if ischema==2
y=b*dt
end

endfunction


function y=rho_inlet(x)

if iinlet==0 then

if x > 0.2 & x < 0.5 then

y=1

else

y=0

end

end

endfunction

function y=rho_ex(x,y)

y=rho_inlet(x-(a/b)*y)
 
endfunction
