		epsilon=1E-6;
		
		a1max=maxi(abs(a1_phys(rho)));
		a1max=max(a1max,epsilon);
		
		a2max=maxi(abs(a2_phys(rho)));
		a2max=max(a2max,epsilon);
		
   dt = cfl * min(dx/a1max, dy/a2max );
