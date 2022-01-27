function y=phi(x)
  
  if k_NL==0 then
  y=1.
  end
  
  if k_NL==1 then
  y=((Ti-T1)*x/T0+T1/T0)**r
  end
  
endfunction
