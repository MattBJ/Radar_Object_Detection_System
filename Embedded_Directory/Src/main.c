/**
  ******************************************************************************
  * @file    Templates/Src/main.c 
  * @author  MCD Application Team
  * @brief   Main program body
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2017 STMicroelectronics.
  * All rights reserved.</center></h2>
  *
  * This software component is licensed by ST under BSD 3-Clause license,
  * the "License"; You may not use this file except in compliance with the
  * License. You may obtain a copy of the License at:
  *                        opensource.org/licenses/BSD-3-Clause
  *
  ******************************************************************************
  */

/* Includes ------------------------------------------------------------------*/
//#include <stm32h743xx.h> // do it first.. fpu check reasons
#include "main.h"

/* DSP Includes */
//#define ARM_MATH_CM7 // for the arm math i guess.. need a better way of doing this
#include <arm_math.h>
#include <arm_const_structs.h>
#include <arm_common_tables.h>

/* Going to maybe include CMSIS DSP through project properties ?
    -> REMOVING FROM PREPROCESSOR LOOKUP DIRECTORIES********************/

#define BLOCK_SIZE      ((uint8_t)32)
#define SML_BLOCK_SIZE  ((uint8_t)4)
#define BLOCK_SIZE_3    ((uint32_t)3)
//#define __FPU_PRESENT   (1u)

/** @addtogroup STM32H7xx_HAL_Examples
  * @{
  */

/** @addtogroup Templates
  * @{
  */

/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
static void SystemClock_Config(void);
static void Error_Handler(void);
static void CPU_CACHE_Enable(void);

/* Private functions ---------------------------------------------------------*/

/**
  * @brief  Main program
  * @param  None
  * @retval None
  */
int main(void)
{
  /* This project template calls firstly CPU_CACHE_Enable() function in order enable the CPU Cache.
     These functions are provided as template implementation that User may integrate in his application.
  */ 


  /* Enable the CPU Cache */
  CPU_CACHE_Enable();

  /* STM32H7xx HAL library initialization:
       - Configure the Systick to generate an interrupt each 1 msec
       - Set NVIC Group Priority to 4
       - Low Level Initialization
     */
  HAL_Init();

  /* Configure the system clock to 400 MHz */
  SystemClock_Config();


  /* Add your application code here */
  
  /* Beginning of DSP code
      Going to try and learn the following:
        - SUPPORT FUNCTIONS
        - Matrix math
        - Variance
        - Standard deviation
        - Use fast math (sqrt, probably for covar)
        - Covariance?
  */
  
  /* First: enable FPU (on reset is disabled)*/
  
  *((uint32_t *)0xE000ED88) |= 0x00F00000; // set bits 20-23 to enable CP10 and CP11 coprocessors
  __DSB(); // assembly instructions
  __ISB(); // RESETING PIPELINE NOW THAT FPU IS ENABLED

  q15_t ex_vec[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32};

  /* copy a vector */
  //const size_t n = sizeof(ex_vec)/sizeof(ex_vec[0]);
  q15_t copy_vec[BLOCK_SIZE];
  
  //q15_t *src = &ex_vec[0], *dst = &copy_vec[0];
  
  arm_copy_q15(&ex_vec[0],&copy_vec[0],BLOCK_SIZE);
  
  /* Copy function works */
  
  /* Let's try copy function again, with float*/
  
  float32_t ex1_vec[] = {0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,
                        1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,
                        2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1};
  float32_t copy1_vec[BLOCK_SIZE];
  
  arm_copy_f32(&ex1_vec[0],&copy1_vec[0],BLOCK_SIZE);
  
  /* Floating point 32, max in this DSP library, works! */
  
  /* NOTE: have to edit literally every C file that uses FPU to include the fucking __FPU_PRESENT header file*/
                    /* -- <stm32h743xx.h>*/
  
  /* Other support functions are conversions b/n floating and standard int types.. whatev*/
  
  /* NEXT: 'Basic math' -- vector math! */
  
  q15_t ex2_vec[] = {10,4,7,3};
  q15_t ex3_vec[] = {2,7,9,11};
  
  /* First: Dot product */
  
  q63_t dot_prod;
  
  arm_dot_prod_q15(&ex2_vec[0],&ex3_vec[0],SML_BLOCK_SIZE,&dot_prod);
  
  q15_t ex4_vec[] = {1, 2, 3, 4};
  q15_t ex5_vec[] = {5, 6,7,8};
  
  q15_t vec_prod[BLOCK_SIZE_3];
  /* Vector product needs to be debugged. Look up calculator and check*/
  arm_mult_q15(&ex4_vec[0],&ex5_vec[0],&vec_prod[0],SML_BLOCK_SIZE);
  
  /* Done with vector product for now.. onto matrix stuff! */
  
  /* Matrix initialization: */
  
  arm_matrix_instance_q15 mat_ex0;
  
  q15_t array_to_mat[5][5] =  { {1,2,3,4,5},
                                {6,7,8,9,10},
                                {11,12,13,14,15},
                                {16,17,18,19,20},
                                {21,22,23,24,25} };
  
  arm_mat_init_q15(&mat_ex0,5,5,&array_to_mat[0][0]);
  
  /* Works! */
  arm_matrix_instance_q15 mat_ex1, mat_ex2, mat_out0;
  q15_t tmp; // 'points to the array for storing intermediate results' 'Unused'
  
  q15_t sml_ar_to_mat[2][2] = { {1,2},
                                {3,4} };
  
  const q15_t sml_ar[4] = {1,2,3,4};
  
  
  arm_mat_init_q15(&mat_ex1,2,2,(q15_t *)sml_ar);
  arm_mat_init_q15(&mat_ex2,2,2,(q15_t *)sml_ar);
  
  volatile arm_status mm_status;
//  mm_status = arm_mat_add_q15(&mat_ex1,&mat_ex2,&mat_out0);
  
//  if(mm_status == ARM_MATH_SIZE_MISMATCH)
//    return 0;
  
  mm_status = arm_mat_mult_q15(&mat_ex1,&mat_ex2,&mat_out0,&tmp);
  if(mm_status == ARM_MATH_SIZE_MISMATCH)
    return 0;
  
  
  BSP_LED_Init(LED1);
  BSP_LED_Init(LED2);
  BSP_LED_Init(LED3);

  /* Infinite loop */
  while (1)
  {
    BSP_LED_Off(LED3);
    BSP_LED_On(LED1);
    HAL_Delay(100);
    BSP_LED_Off(LED1);
    BSP_LED_On(LED2);
    HAL_Delay(100);
    BSP_LED_Off(LED2);
    BSP_LED_On(LED3);
    HAL_Delay(100);
    
  }
}

/**
  * @brief  System Clock Configuration
  *         The system Clock is configured as follow : 
  *            System Clock source            = PLL (HSE BYPASS)
  *            SYSCLK(Hz)                     = 400000000 (CPU Clock)
  *            HCLK(Hz)                       = 200000000 (AXI and AHBs Clock)
  *            AHB Prescaler                  = 2
  *            D1 APB3 Prescaler              = 2 (APB3 Clock  100MHz)
  *            D2 APB1 Prescaler              = 2 (APB1 Clock  100MHz)
  *            D2 APB2 Prescaler              = 2 (APB2 Clock  100MHz)
  *            D3 APB4 Prescaler              = 2 (APB4 Clock  100MHz)
  *            HSE Frequency(Hz)              = 8000000
  *            PLL_M                          = 4
  *            PLL_N                          = 400
  *            PLL_P                          = 2
  *            PLL_Q                          = 4
  *            PLL_R                          = 2
  *            VDD(V)                         = 3.3
  *            Flash Latency(WS)              = 4
  * @param  None
  * @retval None
  */
static void SystemClock_Config(void)
{
  RCC_ClkInitTypeDef RCC_ClkInitStruct;
  RCC_OscInitTypeDef RCC_OscInitStruct;
  HAL_StatusTypeDef ret = HAL_OK;
  
  /*!< Supply configuration update enable */
  HAL_PWREx_ConfigSupply(PWR_LDO_SUPPLY);

  /* The voltage scaling allows optimizing the power consumption when the device is
     clocked below the maximum system frequency, to update the voltage scaling value
     regarding system frequency refer to product datasheet.  */
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  while(!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}
  
  /* Enable HSE Oscillator and activate PLL with HSE as source */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_BYPASS;
  RCC_OscInitStruct.HSIState = RCC_HSI_OFF;
  RCC_OscInitStruct.CSIState = RCC_CSI_OFF;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;

  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLN = 400;
  RCC_OscInitStruct.PLL.PLLFRACN = 0;
  RCC_OscInitStruct.PLL.PLLP = 2;
  RCC_OscInitStruct.PLL.PLLR = 2;
  RCC_OscInitStruct.PLL.PLLQ = 4;

  RCC_OscInitStruct.PLL.PLLVCOSEL = RCC_PLL1VCOWIDE;
  RCC_OscInitStruct.PLL.PLLRGE = RCC_PLL1VCIRANGE_1;
  ret = HAL_RCC_OscConfig(&RCC_OscInitStruct);
  if(ret != HAL_OK)
  {
    Error_Handler();
  }
  
/* Select PLL as system clock source and configure  bus clocks dividers */
  RCC_ClkInitStruct.ClockType = (RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_D1PCLK1 | RCC_CLOCKTYPE_PCLK1 | \
                                 RCC_CLOCKTYPE_PCLK2  | RCC_CLOCKTYPE_D3PCLK1);

  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.SYSCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB3CLKDivider = RCC_APB3_DIV2;  
  RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV2; 
  RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV2; 
  RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV2; 
  ret = HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4);
  if(ret != HAL_OK)
  {
    Error_Handler();
  }

/*
  Note : The activation of the I/O Compensation Cell is recommended with communication  interfaces
          (GPIO, SPI, FMC, QSPI ...)  when  operating at  high frequencies(please refer to product datasheet)       
          The I/O Compensation Cell activation  procedure requires :
        - The activation of the CSI clock
        - The activation of the SYSCFG clock
        - Enabling the I/O Compensation Cell : setting bit[0] of register SYSCFG_CCCSR
  
          To do this please uncomment the following code 
*/
 
  /*  
  __HAL_RCC_CSI_ENABLE() ;
  
  __HAL_RCC_SYSCFG_CLK_ENABLE() ;
  
  HAL_EnableCompensationCell();
  */ 
	
}

/**
  * @brief  This function is executed in case of error occurrence.
  * @param  None
  * @retval None
  */
static void Error_Handler(void)
{
  /* User may add here some code to deal with this error */
  while(1)
  {
  }
}

/**
  * @brief  CPU L1-Cache enable.
  * @param  None
  * @retval None
  */
static void CPU_CACHE_Enable(void)
{
  /* Enable I-Cache */
  SCB_EnableICache();

  /* Enable D-Cache */
  SCB_EnableDCache();
}

#ifdef  USE_FULL_ASSERT

/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t* file, uint32_t line)
{ 
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */

  /* Infinite loop */
  while (1)
  {
  }
}
#endif

/**
  * @}
  */

/**
  * @}
  */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
