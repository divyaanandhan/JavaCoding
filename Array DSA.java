class Solution {
    public int[] twoSum(int[] nums, int target) {
        int output []= new int[2];
        int n= nums.length;
        for (int i = 0; i < n; i++) {
            for (int j = i+1 ; j < n; j++) {
                if (nums[i] + nums[j] == target) {
                    output[0]=i;
                    output[1]=j;
                    return output;
                }
            }
        }

        return output;
        
    }
}
//another approach
class Solution {
    public int[] twoSum(int[] nums, int target) {
        int next = 1;
        while(true) {
            for (int i = 0; i < nums.length - next; i++) {
                if (nums[i] + nums[i + next] == target)
                    return new int[]{ i, next + i };
            }
            next++;
        }
    }
}




class Solution {
    public void sortColors(int[] nums) {
        int countZero =0;//2
        int countOne=0;//2
        int countTwo=0;//2
        for(int i=0;i<nums.length;i++){
            if(nums[i]==0)
                countZero++;            
            else if(nums[i]==1)           
                countOne++;           
            else           
                countTwo++;                    
        }
        for(int i=0;i<countZero;i++)
            nums[i]=0;
        for(int i=countZero;i<countOne+countZero;i++)
            nums[i]=1;
        for(int i=countOne+countZero;i<nums.length;i++)
            nums[i]=2;
    }
}

class Solution {
    public void sortColors(int[] nums) {
        int z = 0, o = 0, t = 0, i = 0;
        for(int n : nums) {
            if(n == 0) {
                z++;
            } else if(n == 1) {
                o++;
            }
        }
        t = nums.length - z - o;
        Arrays.fill(nums, 0, z, 0);
        Arrays.fill(nums, z, z + o, 1);
        Arrays.fill(nums, z + o, nums.length, 2);
    }
}

// class Solution {
//     public int majorityElement(int[] nums) {
//          int n = nums.length;//7
//          int halfLength = n/2;//3
//          int count = 1;
//          int j=0;
//          int majorityElement = nums[0];
         
//          for( int i=0;i<n-1;i++){ 
//            int num = nums[i];       
//             if(nums[i]==nums[j+1])
//                  count++;//4
//             j++;
//             // else{
//             //     count--;
//             // }
//             if(count>halfLength)
//                 majorityElement = num;
//                 break;
//          } 
//         return majorityElement;   
       
//     }
// }







class Solution {
    public int majorityElement(int[] nums) {
        int n = nums.length;
        int halfLength = n / 2;
        
        int count = 1;
        int majorityElement = nums[0];
        
        for (int i = 1; i < n; i++) { // Start the loop from index 1, as we have already initialized count and majorityElement with the first element
            if (nums[i] == majorityElement) {
                count++;
            } else {
                count--;
            }
            
            if (count == 0) { // If count becomes 0, update the majority element and reset the count
                majorityElement = nums[i];
                count = 1;
            }
        }
        
        return majorityElement;
    }
}





class Solution{
    //Function to find the leaders in the array.
    static ArrayList<Integer> leaders(int arr[], int n){
        ArrayList<Integer> output= new ArrayList<Integer>();
        int leader = arr[n-1];
        output.add(leader);
        for(int i=n-2;i>=0;i--){
            if(arr[i] >=leader){
                leader = arr[i];
                output.add(leader);
                

            }
            
        }
        Collections.reverse(output);
        return output;
    }
}









class Solution {
    public int[] rearrangeArray(int[] nums) {
        int[] result = new int[nums.length];
        int posIndex = 0;
        int negIndex = 1;

        for (int num : nums) {
            if (num < 0) {
                result[negIndex] = num;
                negIndex += 2;
            } else {
                result[posIndex] = num;
                posIndex += 2;
            }
        }

        return result;
    }
}



class Solution {
    public int maxProfit(int[] prices) {
        int minPrice = Integer.MAX_VALUE;
        int maxProfit = 0;
        
        for (int i = 0; i < prices.length; i++) {
            if (prices[i] < minPrice) {
                minPrice = prices[i];
            } else if (prices[i] - minPrice > maxProfit) {
                maxProfit = prices[i] - minPrice;
            }
        }
        
        return maxProfit;
    }
}








import java.util.ArrayList;
import java.util.List;

class Solution {
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> ans = new ArrayList<>();
        int n = matrix.length;
        if (n == 0) {
            return ans;
        }
        int m = matrix[0].length;
        int top = 0;
        int left = 0;
        int right = m - 1;
        int bottom = n - 1;

        while (top <= bottom && left <= right) {
            // Traverse top row
            for (int i = left; i <= right; i++) {
                ans.add(matrix[top][i]);
            }
            top++;

            // Traverse right column
            for (int i = top; i <= bottom; i++) {
                ans.add(matrix[i][right]);
            }
            right--;

            // Check if there are remaining rows and columns
            if (top <= bottom) {
                // Traverse bottom row
                for (int i = right; i >= left; i--) {
                    ans.add(matrix[bottom][i]);
                }
                bottom--;
            }

            if (left <= right) {
                // Traverse left column
                for (int i = bottom; i >= top; i--) {
                    ans.add(matrix[i][left]);
                }
                left++;
            }
        }
        return ans;
    }
}










class Solution 
{
    public void rotate(int[][] matrix) 
    {
        for(int i=0;i<matrix.length;i++)
        {
            for(int j=i;j<matrix[0].length;j++)
            {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i]=temp;
            }
        }
        for(int i=0;i<matrix.length;i++)
        {
            for (int j = 0; j < matrix.length/2; j++) 
            {
                int temp = 0;
                temp = matrix[i][j];
                matrix[i][j] = matrix[i][matrix.length - 1 - j];
                matrix[i][matrix.length - 1 - j] = temp;

            }
        }
    }
}



import java.util.HashSet;
import java.util.Set;

class Solution {
    public int longestConsecutive(int[] nums) {
        Set<Integer> numSet = new HashSet<>();
        for (int num : nums) {
            numSet.add(num);
        }
        
        int longestNum = 0;
        for (int num : nums) {
            if (!numSet.contains(num - 1)) {
                int currentNum = num;
                int currentStreak = 1;
                
                while (numSet.contains(currentNum + 1)) {
                    currentNum += 1;
                    currentStreak += 1;
                }
                
                longestNum = Math.max(longestNum, currentStreak);
            }
        }
        
        return longestNum;
    }
}








































// import java.util.HashSet;
// import java.util.Set;
// class Solution {
//     public int longestConsecutive(int[] nums) 
//     {
//         Set<Integer>numSet = new HashSet<Integer>();
//         for(int num:nums)
//         {
//             numSet.add(num);
//         }
//         int longestNum=0;
//         for(int i=0;i<nums.length;i++)
//         {
//             if(!numSet.contains(nums[i]-1))
//             {
//                 int currentNum =i;
//                 longestNum =1;
//                 while(numSet.contains(nums[i]+1))
//                 {
//                 currentNum+=1;
//                 longestNum+=1;
//                 }
//                 longestNum=Math.max(longestNum,currentNum);
//             }
//         }
//         return longestNum;

//     }
// }

class Solution {
    public int maxSubArray(int[] nums) {
        int sum =0;
        int max = Integer.MIN_VALUE;
        for(int i=0;i<nums.length;i++){
            sum += nums[i];
            if(sum>max){
                max=sum;
            }
            if(sum<0){
                sum = 0;
            }
        }
        return max;
    }
}




//User function Template for Java

class Solution {
    
    public static long pairWithMaxSum(long arr[], long N)
    {
        
        long max = Long.MIN_VALUE;
        long sum =0;
        for(int i=1;i<N;i++){
            sum = Math.max(arr[i]+arr[i-1],sum);
        }
        return sum;
    }
}



//User function Template for Java

class Solution
{
    // Function for finding maximum and value pair
    public static int lenOfLongSubarr (int A[], int N, int K)
    {
       HashMap<Integer, Integer> map = new HashMap<>();
        int longest = 0;
        int sum = 0;
        
        for (int i = 0; i < N; i++) {
            sum += A[i];
            
            if (sum == K) {
                longest = i + 1;
            }
            
            if (map.containsKey(sum - K)) {
                longest = Math.max(longest, i - map.get(sum - K));
            }
            
            if (!map.containsKey(sum)) {
                map.put(sum, i);
            }
        }
        
        return longest;

    }
}   
 








class Solution {
    public List<Integer> majorityElement(int[] nums) {
        List<Integer> ls = new ArrayList<>();
        int n=nums.length;
        int N= n/3;
        int c1=0;
        int c2=0;
        int el1 = Integer.MIN_VALUE;
        int el2= Integer.MIN_VALUE;
        for(int i=0;i<n;i++){
            if(c1==0 && el2!=nums[i]){
                c1=1;
                el1=nums[i];
            }
            else if(c2==0 && el1!=nums[i]){
                c2=1;
                el2=nums[i];
            }
            else if(el1==nums[i]){
                c1++;
                // ls.add(nums[i]);
            }
            else if(el2==nums[i]){
                c2++;  
            }
            else
                c1--;
                c2--;
        }
        c1 = 0;
        c2 = 0;
        for (int num : nums) {
            if (num == el1) {
                c1++;
            } else if (num == el2) {
                c2++;
            }
        }
        if (c1 > N) {
            ls.add(el1);
        }
        if (c2 > N) {
            ls.add(el2);
        }
        
        return ls;
    }
}


class Solution 
{
    public int maxProduct(int[] nums)
     {
        int prod1 = nums[0];
        int prod2 = nums[0];
        int result = nums[0];
        for(int i=1;i<nums.length;i++)
        {
            int temp = Math.max(nums[i],Math.max(prod1*nums[i],prod2*nums[i]));
            prod2 = Math.min(nums[i],Math.min(prod1*nums[i],prod2*nums[i]));
            prod1 = temp;
            result = Math.max(result,prod1);
        }
        return result;
    }
}



//User function Template for Java

class Solution
{
    //Function to merge the arrays.
    public static void merge(long arr1[], long arr2[], int n, int m) 
    {
        // code here 
        long[] arr3 = new long[n+m];
        int i = n-1;
        int j = 0;
        while(i>=0 && j<m){
            if(arr1[i]>arr2[j]){
                long swap = arr1[i];
                arr1[i]=arr2[j];
                arr2[j]=swap;
                i--;
                j++;
            }
            else{
                break;
            }
            
            
        }
        Arrays.sort(arr1);
        Arrays.sort(arr2);
    
        
    }
}


import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> ans = new ArrayList<>();
        Arrays.sort(nums);

        for (int i = 0; i < nums.length - 2; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }

            int left = i + 1;
            int right = nums.length - 1;
            int target = -nums[i];

            while (left < right) {
                int sum = nums[left] + nums[right];

                if (sum == target) {
                    ans.add(Arrays.asList(nums[i], nums[left], nums[right]));

                    while (left < right && nums[left] == nums[left + 1]) {
                        left++;
                    }
                    while (left < right && nums[right] == nums[right - 1]) {
                        right--;
                    }

                    left++;
                    right--;
                } else if (sum < target) {
                    left++;
                } else {
                    right--;
                }
            }
        }

        return ans;
    }
}














































































// class Solution {
//     public List<List<Integer>> threeSum(int[] nums) {
//         List<List<Integer>> ans = new ArrayList<>();

//         int j =0;
//         int k=0;
//         for( int i=0;i<nums.length;i++){
//             if(i>0&&nums[i]==nums[i-1])continue;
//             j=i+1;
//             k=nums.length-1;
            
//             while(j<k){
//                 if(nums[i]+nums[j]+nums[k]<0)j++;
//                 if(nums[i]+nums[j]+nums[k]>0)k--;
//                 if(nums[i]+nums[j]+nums[k]==0){
//                     ans.add(Arrays.asList(nums[i], nums[j], nums[k]));
//                      while(j<k){
//                          if(j==j-1)j++;
//                          if(k==k+1)k--;
//                     }
//                     j++;
//                     k--;
//                 }
//             }
           
//         }
        
//         return ans;
        
//     }
// }















class Solution
{
    void merge(int arr[], int l, int m, int r)
    {
         // Your code here
         ArrayList<Integer> temp = new ArrayList<>();
         int left = l;
         int right = m+1;
         while(left<=m&&right<=r){
             if(arr[left]<=arr[right]){
                 temp.add(arr[left]);
                 left++;
             }
             else{
                 temp.add(arr[right]);
                 right++;
             }
         }
         while(left<=m){
             temp.add(arr[left]);
             left++;
         }
         while(right<=r)
          {
                 temp.add(arr[right]);
                 right++;
          }
         for(int i=l;i<=r;i++){
            arr[i] = temp.get(i-l);
          }
        }
		
    void mergeSort(int arr[], int l, int r)
    {
        if (l >= r)
            return;
            
        int m = (l + r) / 2;
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}




// //User function Template for Java



class Solution {
    // arr[]: Input Array
    // N : Size of the Array arr[]
    // Function to count inversions in the array.
    static long inversionCount(long arr[], long N) {
        return mergeSort(arr, 0, N - 1);
    }

    static long mergeSort(long arr[], long l, long r) {
        long count = 0;
        if (l < r) {
            long m = (l + r) / 2;
            count += mergeSort(arr, l, m);
            count += mergeSort(arr, m + 1, r);
            count += merge(arr, l, m, r);
        }
        return count;
    }

    static long merge(long arr[], long l, long m, long r) {
        ArrayList<Long> temp = new ArrayList<>();
        long count = 0;
        long left = l;
        long right = m + 1;

        while (left <= m && right <= r) {
            if (arr[(int) left] <= arr[(int) right]) {
                temp.add(arr[(int) left]);
                left++;
            } else {
                temp.add(arr[(int) right]);
                count += m - left + 1;
                right++;
            }
        }

        while (left <= m) {
            temp.add(arr[(int) left]);
            left++;
        }
        while (right <= r) {
            temp.add(arr[(int) right]);
            right++;
        }

        for (int i = 0; i < temp.size(); i++) {
            arr[(int) (l + i)] = temp.get(i);
        }

        return count;
    }

    
}


class Solution {
    public List<List<Integer>> fourSum(int[] nums, int target) {
        Arrays.sort(nums);
        return kSum(nums, target, 0, 4);
    }
	
    public List<List<Integer>> kSum(int[] nums, long target, int start, int k) {
        List<List<Integer>> res = new ArrayList<>();

        // If we have run out of numbers to add, return res.
        if (start == nums.length) {
            return res;
        }
        
        // There are k remaining values to add to the sum. The 
        // average of these values is at least target / k.
        long average_value = target / k;
        
        // We cannot obtain a sum of target if the smallest value
        // in nums is greater than target / k or if the largest 
        // value in nums is smaller than target / k.
        if  (nums[start] > average_value || average_value > nums[nums.length - 1]) {
            return res;
        }
        
        if (k == 2) {
            return twoSum(nums, target, start);
        }
    
        for (int i = start; i < nums.length; ++i) {
            if (i == start || nums[i - 1] != nums[i]) {
                for (List<Integer> subset : kSum(nums, target - nums[i], i + 1, k - 1)) {
                    res.add(new ArrayList<>(Arrays.asList(nums[i])));
                    res.get(res.size() - 1).addAll(subset);
                }
            }
        }
    
        return res;
    }
	
    public List<List<Integer>> twoSum(int[] nums, long target, int start) {
        List<List<Integer>> res = new ArrayList<>();
        int lo = start, hi = nums.length - 1;
    
        while (lo < hi) {
            int currSum = nums[lo] + nums[hi];
            if (currSum < target || (lo > start && nums[lo] == nums[lo - 1])) {
                ++lo;
            } else if (currSum > target || (hi < nums.length - 1 && nums[hi] == nums[hi + 1])) {
                --hi;
            } else {
                res.add(Arrays.asList(nums[lo++], nums[hi--]));
            }
        }
                                                          
        return res;
    }
}



















































// class Solution {
//     public List<List<Integer>> fourSum(int[] nums, int target) {
//         List<List<Integer>> ans = new ArrayList<>();
//         if (nums.length < 4) {
//             return ans;
//         }
//         Arrays.sort(nums);

//         for (int i = 0; i < nums.length - 3; i++) {
//             if (i > 0 && nums[i] == nums[i - 1]) {
//                 continue;
//             }
//             for (int j = i + 1; j < nums.length - 2; j++) {
//                 if (j > i + 1 && nums[j] == nums[j - 1]) {
//                     continue;
//                 }

//                 int left = j + 1;
//                 int right = nums.length - 1;

//                 while (left < right) {
//                     int sum = nums[i] + nums[j] + nums[left] + nums[right];
//                     if (sum == target) {
//                         ans.add(Arrays.asList(nums[i], nums[j], nums[left], nums[right]));

//                         while (left < right && nums[left] == nums[left + 1]) {
//                             left++;
//                         }
//                         left++;
//                         while (left < right && nums[right] == nums[right - 1]) {
//                             right--;
//                         }
//                         right--;
//                     } else if (sum < target) {
//                         left++;
//                     } else {
//                         right--;
//                     }
//                 }
//             }
//         }

//         return ans;
//     }
// }






// User function Template for Java

// class Solve {
//     int[] findTwoElement(int arr[], int n) {
//         int[] result = new int[2];
//         int SN = (n * (n + 1)) / 2;
//         int S2N = (n * (n + 1) * (2 * n + 1)) / 6;
//         int S = 0;

//         for (int i = 0; i < n; i++) {
//             S += arr[i];
//             int S2 = arr[i] * arr[i];
//             int Sum = S - SN; // X - Y
//             int Square = S2 - S2N; // X^2 - Y^2
//             Square = Sum / Square;
//             int rep = (Sum + Square) / 2;
//             int miss = Sum - rep;

//             result[0] = rep;
//             result[1] = miss;
//         }

//         return result;
//     }
// }












class Solve {
    int[] findTwoElement(int arr[], int n) {
        int[] result = new int[2];
        int xor = 0;
        for (int i = 0; i < n; i++) {
            xor ^= arr[i];
            xor ^= (i + 1);
        }

        int rightmostSetBit = xor & ~(xor - 1);
        int num1 = 0, num2 = 0;

        for (int i = 0; i < n; i++) {
            if ((arr[i] & rightmostSetBit) != 0)
                num1 ^= arr[i];
            else
                num2 ^= arr[i];
        }

        for (int i = 1; i <= n; i++) {
            if ((i & rightmostSetBit) != 0)
                num1 ^= i;
            else
                num2 ^= i;
        }

        for (int i = 0; i < n; i++) {
            if (arr[i] == num1) {
                result[0] = num1;
                result[1] = num2;
                break;
            } else if (arr[i] == num2) {
                result[0] = num2;
                result[1] = num1;
                break;
            }
        }

        return result;
    }
}




// User function Template for Java

// import java.util.ArrayList;

// class Solution {
//     ArrayList<Long> nthRowOfPascalTriangle(int n) {
//         ArrayList<Long> myList = new ArrayList<>();
//         long res = 1;
//         myList.add(res);

//         for (int i = 1; i < n; i++) {
//             res = res * (n - i) / i;
//             myList.add(res);
//         }

//         return myList;
//     }
// }
// Back-end complete function Template for Java

class Solution {
  

    ArrayList<Long> nthRowOfPascalTriangle(int n) {
        if(n==1)
        {
                    ArrayList<Long> v = new ArrayList<>();
                    v.add(1l);
                    return v;
        }
        final long MOD = 1000_000_007;
        
        ArrayList<Long> v = new ArrayList<>();
        ArrayList<Long> tans = nthRowOfPascalTriangle(n-1);
        v.add(1l);
        long c = 1;
        for (int i = 1; i < n-1; i++) {
            v.add((tans.get(i)+tans.get(i-1))%MOD);
            //c = ((c % MOD * (n - i)) % MOD * mod.get(i)) % MOD;
        }
        v.add(1l);
        return v;
    }
}





class Solution {
  public List<List<Integer>> generate(int numRows) {
    List<List<Integer>> ans = new ArrayList<>();

    for (int i = 0; i < numRows; ++i) {
      Integer[] temp = new Integer[i + 1];
      Arrays.fill(temp, 1);
      ans.add(Arrays.asList(temp));
    }

    for (int i = 2; i < numRows; ++i)
      for (int j = 1; j < ans.get(i).size() - 1; ++j)
        ans.get(i).set(j, ans.get(i - 1).get(j - 1) + ans.get(i - 1).get(j));

    return ans;
  }
}













// import java.util.ArrayList;
// import java.util.List;
// class Solution 
// {
//     public List<List<Integer>> generate(int numRows) 
//     {
//             long ans = 1;
//             List<Integer> ansRow = new ArrayList<>();
//             ansRow.add(1); //inserting the 1st element

//             //calculate the rest of the elements:
//             for (int col = 1; col < numRows; col++)
//              {
//                 ans = ans * (numRows - col);
//                 ans = ans / col;
//                 ansRow.add((int)ans);
//             }
//             return ansRow;
//     }

//     public List<List<Integer>> pascalTriangle(int n)
//     {
//             List<List<Integer>> ans = new ArrayList<>();

//             //store the entire pascal's triangle:
//             for (int numRows = 1; numRows <= n; numRows++)
//              {
//                 ans.add(generate(numRows));
//             }
//             return ans;
//     }
    
// }
   


    //     long res = 1;
    //     List<List<Integer>> ansRows = new ArrayList<>();
    //     ansRows.add(1);
    //     for(int col = 1;col<numRows;col++){
    //         res = res*(numRows-col);
    //         res = res/col;
    //         ansRows.add((int)res);
    //     }
    //     return ansRows;

    // }
    // public List<List<Integer>> pascalsTriangle(int n)
    // {
    //     List<List<Integer>> res = new ArrayList<>();
    //     for(int numRows = 1;numRows<=n;numRows++){
    //         res.add(generate(numRows));
    //     }
    //     return res;

    // }
// }













public class Solution {

    public static int subsetXOR(int[] arr, int N, int K) {
        int count = 0;
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < (1 << N); i++) {
            int xr = 0;
            for (int j = 0; j < N; j++) {
                if ((i & (1 << j)) != 0) {
                    xr = xr ^ arr[j];
                }
            }
            if (!map.containsKey(xr)) {
                map.put(xr, 1);
            } else {
                map.put(xr, map.get(xr) + 1);
            }
        }
        if (map.containsKey(K)) {
            count = map.get(K);
        }
        return count;
    }
}

//Back-end complete function Template for Java

class Solution{
    static int subsetXOR(int arr[], int N, int K) 
    { 
        // Find maximum element in arr[] 
        int max_ele = arr[0]; 
        for (int i=1; i<N; i++) 
          if (arr[i] > max_ele) 
              max_ele = arr[i]; 
        // Maximum possible XOR value 
        int m = 10*max_ele;
            
        int[][] dp = new int[N+1][m+1];
         // The xor of empty subset is 0 
        dp[0][0] = 1;
        // Fill the dp table 
        for(int i=1;i<=N;i++){
            for(int j=0;j<=m;j++){
                dp[i][j] += dp[i-1][j]; 
                if((j^arr[i-1])<=m){
                    dp[i][j] += dp[i-1][j^arr[i-1]];
                }
            }
        }
        //  The answer is the number of subset from set 
        //  arr[0..n-1] having XOR of elements as k 
        return dp[N][K]; 
    } 
}









class Solution
{
	int  select(int arr[], int i)
	{
	    selectionSort(arr,arr.length);
	    return arr[i];
        // code here such that selectionSort() sorts arr[]
	}
	
	void selectionSort(int arr[], int n)
	{
	    //code here
	    for(int i=0;i<n-1;i++)
	    {
	        int mini = i;
	        for(int j=i+1;j<n;j++)
	        {
	            if(arr[j]<arr[mini])
	            {
	                mini =j;
	            }
	        }
	        int temp = arr[i];
	        arr[i] = arr[mini];
	        arr[mini]= temp;
	        
	    }
	}
}

class Solution
{
    //Function to sort the array using bubble sort algorithm.
	public static void bubbleSort(int arr[], int n)
    {
        //code here
        for(int i=n-1;i>=0;i--){
            for(int j=0;j<=i-1;j++){
                if(arr[j]>arr[j+1]){
                    int temp = arr[j];
                    arr[j] = arr[j+1];
                    arr[j+1] = temp;
                }
            }
        }
    }
}

class Solution {
    // Function to sort the array using bubble sort algorithm.
    public static void bubbleSort(int arr[], int n) {
        // code here
        for (int i = n - 1; i >= 0; i--) {
            boolean didSwap = false;
            for (int j = 0; j < i; j++) {
                if (arr[j] > arr[j + 1]) {
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                    didSwap = true;
                }
            }
            if (!didSwap) {
                break;
            }
        }
    }
}



class Solution
{
  static void insert(int arr[],int i){
      Solution sol = new Solution();
      sol.insertionSort(arr,arr.length);

  }
  //Function to sort the array using insertion sort algorithm.
  public void insertionSort(int arr[], int n)
  {
      //code here
      
    for(int i=1;i<n;i++)
    {
        int j = i;
        while(j >0 && arr[j-1]>arr[j])
        {
            int temp = arr[j-1];
            arr[j-1] = arr[j];
            arr[j] = temp;
            j--;
        }
          
    }
      
  }

}


class Solution
{
    //Function to sort an array using quick sort algorithm.
    static void quickSort(int arr[], int low, int high)
    {
        if(low < high){
            int partition = partition(arr,low,high);
            quickSort(arr,low,partition-1);
            quickSort(arr,partition+1,high);
        }
    }
    static int partition(int arr[], int low, int high)
{
    int pivot = arr[low];
    int i = low + 1;  // Start from the element next to the pivot
    int j = high;

    while (i <= j) {
        while (i <= j && arr[i] <= pivot) {
            i++;
        }

        while (i <= j && arr[j] > pivot) {
            j--;
        }

        if (i < j) {
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }

    // Swap the pivot element with the element at index j
    int temp = arr[low];
    arr[low] = arr[j];
    arr[j] = temp;

    return j;
}

}

























































































































































































































































   











































































































































