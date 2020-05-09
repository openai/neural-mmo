raw data: {
   Obs_0: {
      Entity_0: {
         Attribute_0: 1
         Attribute_1: 2
         Attribute_2: 3
         Attribute_3: 4
      }
   }
   Obs_1: {
      Entity_0: {
         Attribute_0: 5
         Attribute_1: 6
         Attribute_2: 7
         Attribute_3: 8
 
      }
      Entity_2: {
         Attribute_0: 9
         Attribute_1: 10
         Attribute_2: 11
         Attribute_3: 12
      }
   }
}


as Dict(Dict(Box(1)))
-> [B={}, N={}, M={}]
This is the current behavior and the return
value of unbatch() for the following two examples

as Repeated(Dict(Dict(Box(1))), max_n=3)
-> [B=2, N=3, M=4]
np.ndarray(
   [[1, 2, 3, 4], [-, -, -, -],    [-, -, -, -]],
   [[5, 6, 7, 8], [9, 10, 11, 12], [-, -, -, -]],
)

as Dict(Repeated(Dict(Box(1)), max_n=4))
-> [B={}, N={}, M=4]
{
   Obs_0: {
      Entity_0: np.ndarray([1, 2, 3, 4])
   }
   Obs_1: {
      Entity_0: np.ndarray([1, 2, 3, 4])
      Entity_2: np.ndarray([9, 10, 11, 12])
   }
}
