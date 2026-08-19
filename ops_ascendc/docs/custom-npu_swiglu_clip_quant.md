# custom.npu\_swiglu\_clip\_quant<a name="ZH-CN_TOPIC_0000001979260729"></a>

## 产品支持情况 <a name="zh-cn_topic_0000001832267082_section14441124184110"></a>
| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 推理系列产品</term>   | √  |

## 功能说明<a name="zh-cn_topic_0000001832267082_section14441124184110"></a>

`SwigluClipQuant`在Swish门控线性单元激活函数后添加clamp和quant操作，并根据不同分组中的clamp缩放因子对输入进行缩放操作，实现x的SwigluClipQuant计算，`SwigluClipQuant`的具体计算公式如下：
$$
\textbf{swiglu}: swigluOut = Swiglu(A)*B
$$

$$
\textbf{clip}: swigluClipOut = Max(Min(swigluOut, Abs(ReduceMax(swigluOut)) * groupAlpha), -Abs(ReduceMax(swigluOut)) * groupAlpha)
$$

$$
\textbf{quant}: y, scale = DynamicQuant(swigluClipOut, groupIndex)
$$

其中，A表示输入x的前半部分，B表示输入x的后半部分。

## 函数原型<a name="zh-cn_topic_0000001832267082_section45077510411"></a>

```
custom.npu_swiglu_clip_quant(Tensor x, Tensor group_index, Tensor group_alpha, *, bool activate_left=False, int quant_mode=1, int clamp_mode=1) -> (Tensor, Tensor)
```

## 参数说明<a name="zh-cn_topic_0000001832267082_section112637109429"></a>

>**说明：**<br> 
>
>- TokensNum、H参数维度含义：TokensNum表示传输的Token数，取值是自然数，H表示嵌入向量的长度，取值>0。
>- groupNum：表示group_index输入的长度，取值>0。

-   **x**（`Tensor`）：必选参数，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为：[TokensNum, H]。
    
-   **group_index**（`Tensor`）：必选参数，不支持非连续，数据格式支持ND，数据类型支持`int64`，要求是1D的Tensor。当前只支持count模式，表示该模式下指定分组的Tokens数（要求非负整数），shape为：[groupNum]。
    
-   **group_alpha**（`Tensor`）：必选参数，不支持非连续，数据格式支持ND，数据类型支持`float32`，要求是1D的Tensor。表示与指定分组的Tokens数对应的clamp缩放因子，shape为：[groupNum]。
    
- <strong>*</strong>：代表其之前的参数是位置相关的，必须按照顺序输入，属于必选参数；其之后的参数是键值对赋值，与位置无关，属于可选参数（不传入会使用默认值）。

-   **activate\_left**（`bool`）：可选参数，表示是否对输入的左半部分做swiglu激活，当值为false时，对输入的右半部分做激活，默认值为False。

-   **quant\_mode**（`int`）：可选参数，表示使用动态量化还是静态量化，0表示静态量化，1表示动态量化。默认值为1，当前仅支持动态量化。

-   **clamp\_mode**（`int`）：可选参数，表示是否在指定分组Tokens数时使用group_alpha参数。默认值为1，表示使用group_alpha参数功能。当前仅支持clamp_mode=1。

## 返回值说明<a name="zh-cn_topic_0000001832267082_section22231435517"></a>

-   **y**（`Tensor`）：公式中的输出y，表示量化后的输出tensor，数据类型支持`int8`。数据格式支持ND。
-   **scale**（`Tensor`）：公式中的输出scale，表示量化scale参数，数据类型支持`float32`。数据格式支持ND。

## 约束说明<a name="zh-cn_topic_0000001832267082_section12345537164214"></a>

-   该接口支持推理场景下使用。
-   该接口支持图模式。
-   该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
-   参数x的H轴有维度大小限制：H ≤ 10496同时64对齐场景，规格不满足场景会进行校验。
-   参数group_index只支持count模式，需要保证group_index总和不超过x的TokensNum维度，否则会出现越界访问。
-   输出y和scale超过group_index总和的部分未进行清理处理，该部分内存为垃圾数据，使用时需要注意影响。
-   参数quant_mode仅支持动态量化场景。

## 调用示例<a name="zh-cn_topic_0000001832267082_section14459801435"></a>

-   详见[test_npu_swiglu_clip_quant.py](../examples/test_npu_swiglu_clip_quant.py)