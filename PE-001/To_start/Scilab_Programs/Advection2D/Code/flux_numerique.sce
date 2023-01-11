         if ischema==1 then
         
         for i=1:imax-1
         for j=2:jmax-1
         q1(i,j)=q1_num(rho(i,j),rho(i+1,j));
         f1num(i,j) = 0.5*(f1_phys(rho(i,j))+f1_phys(rho(i+1,j)))-0.5*q1(i,j).*(rho(i+1,j)-rho(i,j));   
         end
         end
         
        
         for i=2:imax-1
         for j=1:jmax-1
         q2(i,j)=q2_num(rho(i,j),rho(i,j+1));
         f2num(i,j) = 0.5*(f2_phys(rho(i,j))+f2_phys(rho(i,j+1)))-0.5*q2(i,j).*(rho(i,j+1)-rho(i,j));   
         end
         end
         
         end
         
         if ischema==2 then
         
         for i=1:imax-1
         for j=2:jmax-1
         q1(i,j)=q1_num(rho(i,j),rho(i+1,j));
         res1=(f1_phys(rho(i+1,j))-f1_phys(rho(i,j)))/dx + 0.25*(f2_phys(rho(i,j+1))+f2_phys(rho(i+1,j+1))-f2_phys(rho(i,j-1))-f2_phys(rho(i+1,j-1)))/dy;
         f1num(i,j) = 0.5*(f1_phys(rho(i,j))+f1_phys(rho(i+1,j)))-0.5*q1(i,j).*res1;   
         end
         end
         
         for i=2:imax-1
         for j=1:jmax-1
         q2(i,j)=q2_num(rho(i,j),rho(i,j+1));
         res2=(f2_phys(rho(i,j+1))-f2_phys(rho(i,j)))/dy + 0.25*(f1_phys(rho(i+1,j))+f1_phys(rho(i+1,j+1))-f1_phys(rho(i-1,j))-f1_phys(rho(i-1,j+1)))/dx;
         f2num(i,j) = 0.5*(f2_phys(rho(i,j))+f2_phys(rho(i,j+1)))-0.5*q2(i,j).*res2;   
         end
         end

				 end