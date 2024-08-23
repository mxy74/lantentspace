/*
    这个js主要用来绘制热力图概览，以及中间的菜单控制
*/
// 根据设置修改鲁棒性示例（thresholds变量在manage_DNN中定义）
// function change_thresholds() {
//     var lowerBound = parseFloat(document.getElementById("lower_bound").value);
//     var step = parseFloat(document.getElementById("threshold_step").value);
//     // console.log("lowerBound: ", lowerBound)
//     // console.log("step: ", step)
//     for (let i = 0; i < 5; i++) {
//         let num = lowerBound + i * step;
//         let str = num.toString();
//         let dotIndex = str.indexOf('.');
//         thresholds[i + 1] = parseFloat(dotIndex == -1 ? str : str.substring(0, dotIndex + 5)); //只保留四位小数
//     }

//     // console.log("thresholds: ",thresholds)
//     // 标记颜色标尺
//     // document.getElementById('threshold-A-value').innerHTML = (thresholds[0] * 1000).toFixed(1) + " ~ " + (thresholds[1] * 1000).toFixed(1);
//     // document.getElementById('threshold-B-value').innerHTML = (thresholds[1] * 1000).toFixed(1) + " ~ " + (thresholds[2] * 1000).toFixed(1);
//     // document.getElementById('threshold-C-value').innerHTML = (thresholds[2] * 1000).toFixed(1) + " ~ " + (thresholds[3] * 1000).toFixed(1);
//     // document.getElementById('threshold-D-value').innerHTML = (thresholds[3] * 1000).toFixed(1) + " ~ " + (thresholds[4] * 1000).toFixed(1);
//     // document.getElementById('threshold-E-value').innerHTML = (thresholds[4] * 1000).toFixed(1) + " ~ " + (thresholds[5] * 1000).toFixed(1);
//     // document.getElementById('threshold-F-value').innerHTML = " ≥ " + (thresholds[5] * 1000).toFixed(1);

// }


// //将小于0的鲁棒性置0
// function no_zero(robustness) {
//     for (var i = 0; i < robustness.length; i++) {
//         // 这里之所以要来回变换数据类型，是为了防止出错，导致热力图画不出来
//         robustness[i] = parseFloat(robustness[i].toString()) >= 0 ? parseFloat(robustness[i].toString()) : 0
//     }
//     return robustness
// }

// //修改热力图-------------------------------------------------------
// async function change_heatMap() {
//     // 修改热力图图例
//     change_thresholds()
//     //根据比例尺与分辨率确定坐标，以及通过后端确定坐标上的鲁棒性
//     var bins = document.getElementById("resolution").value;
//     // console.log(robustness)
//     //绘制热力图
//     draw_heatMap(bins = bins)
// }

// //动态修改热力图(主要是针对选框放大)-------------------------------------------------------
// async function change_heatMap_transition(x, y, length) {
//     // 修改热力图图例
//     change_thresholds()
//     //根据比例尺与分辨率确定坐标，以及通过后端确定坐标上的鲁棒性
//     var bins = document.getElementById("resolution").value;
//     // console.log(robustness)
//     //绘制热力图
//     draw_heatMap_transition(bins = bins, x = x, y = y, length = length)
// }

// //修改热力图，主要是用来画两个模型之间的热力图差异
// async function change_heatMap_difference() {
//     //根据比例尺与分辨率确定坐标，以及通过后端确定坐标上的鲁棒性
//     var bins = document.getElementById("resolution").value;
//     // console.log(robustness)
//     //绘制热力图
//     draw_heatMap_difference(bins = bins)
// }

// //修改热力图，主要是用来画两个模型之间的热力图差异(针对选框放大)
// async function change_heatMap_difference_transition(x, y, length) {
//     //根据比例尺与分辨率确定坐标，以及通过后端确定坐标上的鲁棒性
//     var bins = document.getElementById("resolution").value;
//     // console.log(robustness)
//     //绘制热力图
//     draw_heatMap_difference_transition(bins = bins, x = x, y = y, length = length)
// }


// //绘制热力图------------------------------------------------
// async function draw_heatMap(bins) {
//     /*
//         bins是热力图‘分辨率’ = bins*bins
//         extent是一个代表热力图坐标范围的字典
//     */
//     // 停止转动加载图标
//     var spinner = d3.selectAll(".spinner-border");
//     spinner.style("display", "none");

//     let contour_scale = heatMap_width / bins;

//     var heatMap_svg = d3.select("#center_heatMap_svg")
//         .attr("width", heatMap_width)
//         .attr("height", heatMap_height)

//     // 显示单选按钮选择的模型鲁棒性
//     const selectedRadioModelButton = document.querySelector('input[name="radio_model"]:checked');
//     const selectedValue = selectedRadioModelButton.value;
//     var robustness = robustness_dict[selectedValue]
//     var rob_expand = no_zero(robustness);

//     // console.log("rob_expand: ", rob_expand)
//     // 计算等值线数据---------------------------------
//     let contours = d3.contours()
//         .size([bins, bins])
//         .thresholds(thresholds)
//         // .smooth(0.5)
//         (rob_expand);
//     console.log("contours: ", contours)
//     //颜色标尺
//     color = ["#fee5d9", "#fcbba1", "#fc9272", "#fb6a4a", "#de2d26", "#a50f15"];
//     // color = ["#fee5d9", "#fcbba1", "#fc9272", "#de2d26", "#a50f15", "#67000d"];
//     // 左侧热力图概览绘制------------------------------
//     selection_contour = heatMap_svg.select("#heatMap_g").selectAll("path")
//         .data(contours, d => d.value)
//         .join("path")
//         .attr("d", d3.geoPath())
//         .attr("transform", "scale(" + contour_scale + ")")
//         .attr("fill", (d, i) => color[i])
//         .attr("stroke", "#fff")
//         .attr("stroke-width", "0.0001px")

//     //绘制x，y轴---------------------------------------
//     // 绘制x轴
//     xAxis = d3.axisBottom(xScale)
//     gX = heatMap_svg.select("#x_g")
//         .call(xAxis);
//     // 绘制y轴
//     yAxis = d3.axisRight(yScale);
//     gY = heatMap_svg.select("#y_g")
//         .call(yAxis);
// }


// //动态绘制热力图(加了个渐变动画，)-----------------------------------------------
// async function draw_heatMap_transition(bins, x, y, length) {
//     /*
//         bins是热力图‘分辨率’ = bins*bins
//         extent是一个代表热力图坐标范围的字典
//     */
//     // 停止转动加载图标
//     var spinner = d3.selectAll(".spinner-border");
//     spinner.style("display", "none");

//     let contour_scale = heatMap_width / bins;

//     var heatMap_svg = d3.select("#center_heatMap_svg")
//         .attr("width", heatMap_width)
//         .attr("height", heatMap_height)


//     // 显示单选按钮选择的模型鲁棒性
//     const selectedRadioModelButton = document.querySelector('input[name="radio_model"]:checked');
//     const selectedValue = selectedRadioModelButton.value;
//     var robustness = robustness_dict[selectedValue]
//     var rob_expand = no_zero(robustness);

//     // console.log("rob_expand: ", rob_expand)
//     // 计算等值线数据---------------------------------
//     let contours = d3.contours()
//         .size([bins, bins])
//         // .smooth(smooth)
//         .thresholds(thresholds)
//         (rob_expand);
//     //颜色标尺
//     color = ["#fee5d9", "#fcbba1", "#fc9272", "#fb6a4a", "#de2d26", "#a50f15"];
//     // 左侧热力图概览绘制------------------------------
//     selection_contour = heatMap_svg.select("#heatMap_g").selectAll("path")
//         .data(contours, d => d.value)
//         .join("path")
//         .attr("d", d3.geoPath())
//         .attr("fill", (d, i) => color[i])
//         .attr("stroke", "#fff")
//         .attr("stroke-width", "0.0001px")
//         .attr("transform", "translate(" + x + "," + y + "), scale(" + Number(length) / Number(bins) + ")") //要写在一起才行
//         .transition()
//         .duration(1000)
//         .attr("transform", "scale(" + contour_scale + ")") //只写scale，translate就默认是(0,0)

//     //绘制x，y轴---------------------------------------
//     // 绘制x轴
//     xAxis = d3.axisBottom(xScale)
//     gX = heatMap_svg.select("#x_g")
//         .call(xAxis);
//     // 绘制y轴
//     yAxis = d3.axisRight(yScale);
//     gY = heatMap_svg.select("#y_g")
//         .call(yAxis);
// }

// //绘制差异热力图
// async function draw_heatMap_difference(bins) {
//     /*
//         bins是热力图‘分辨率’ = bins*bins
//     */
//     // 停止转动加载图标
//     var spinner = d3.selectAll(".spinner-border");
//     spinner.style("display", "none");

//     // 隐藏常规的热力图图例
//     d3.select("#thresholds").style("display", "none")
//     d3.select("#thresholds_diff").style("display", "block")


//     let contour_scale = heatMap_width / bins;

//     var heatMap_svg = d3.select("#center_heatMap_svg")
//         .attr("width", heatMap_width)
//         .attr("height", heatMap_height)

//     // 显示单选按钮选择的模型鲁棒性
//     var robustness1 = robustness_dict["M1"]
//     console.log("差异热力图中的robustness1：", robustness1)
//     var robustness2 = robustness_dict["M2"]
//     console.log("差异热力图中的robustness2：", robustness2)
//     var rob_expand = robustness1.map((val, index) => parseFloat((val - robustness2[index]).toFixed(4)))
//     console.log("差异热力图中的rob_expand：", rob_expand)
//     //找到最大值和最小值
//     var min_value = 1e-5
//     var max_value = -1e5
//     for (var i = 0; i < rob_expand.length; i++) {
//         if (rob_expand[i] < min_value) {
//             min_value = rob_expand[i]
//         }
//         if (rob_expand[i] > max_value) {
//             max_value = rob_expand[i]
//         }
//     }
//     var thresholds_diff = [];
//     const dif_value = (max_value - min_value) / 5;
//     for (let i = 0; i < 6; i++) {
//         thresholds_diff.push(parseFloat((min_value + i * dif_value).toFixed(4)));
//     }
//     console.log("差异热力图中的thresholds_diff：", thresholds_diff)
//     // 标记颜色标尺
//     document.getElementById('threshold-A-value_diff').innerHTML = (thresholds_diff[0] * 1000).toFixed(1) + " ~ " + (thresholds_diff[1] * 1000).toFixed(1);
//     document.getElementById('threshold-B-value_diff').innerHTML = (thresholds_diff[1] * 1000).toFixed(1) + " ~ " + (thresholds_diff[2] * 1000).toFixed(1);
//     document.getElementById('threshold-C-value_diff').innerHTML = (thresholds_diff[2] * 1000).toFixed(1) + " ~ " + (thresholds_diff[3] * 1000).toFixed(1);
//     document.getElementById('threshold-D-value_diff').innerHTML = (thresholds_diff[3] * 1000).toFixed(1) + " ~ " + (thresholds_diff[4] * 1000).toFixed(1);
//     document.getElementById('threshold-E-value_diff').innerHTML = (thresholds_diff[4] * 1000).toFixed(1) + " ~ " + (thresholds_diff[5] * 1000).toFixed(1);
//     document.getElementById('threshold-F-value_diff').innerHTML = " ≥ " + (thresholds_diff[5] * 1000).toFixed(1);

//     // console.log("rob_expand: ", rob_expand)
//     // 计算等值线数据---------------------------------
//     let contours = d3.contours()
//         .size([bins, bins])
//         .thresholds(thresholds_diff)
//         // .smooth(0.5)
//         (rob_expand);
//     console.log("差异热力图中的contours: ", contours)
//     //颜色标尺
//     color = ["#005ab5", "#0080ff", "#84c1ff", "#fee5d9", "#fcbba1", "#fc9272"];
//     // color = ["#fee5d9", "#fcbba1", "#fc9272", "#de2d26", "#a50f15", "#67000d"];
//     // 左侧热力图概览绘制------------------------------
//     selection_contour = heatMap_svg.select("#heatMap_g").selectAll("path")
//         .data(contours, d => d.value)
//         .join("path")
//         .attr("d", d3.geoPath())
//         .attr("transform", "scale(" + contour_scale + ")")
//         .attr("fill", (d, i) => color[i])
//         .attr("stroke", "#fff")
//         .attr("stroke-width", "0.0001px")

//     //绘制x，y轴---------------------------------------
//     // 绘制x轴
//     xAxis = d3.axisBottom(xScale)
//     gX = heatMap_svg.select("#x_g")
//         .call(xAxis);
//     // 绘制y轴
//     yAxis = d3.axisRight(yScale);
//     gY = heatMap_svg.select("#y_g")
//         .call(yAxis);
// }

// //动态绘制差异热力图
// async function draw_heatMap_difference_transition(bins, x, y, length) {
//     /*
//         bins是热力图‘分辨率’ = bins*bins
//         extent是一个代表热力图坐标范围的字典
//     */
//     // 停止转动加载图标
//     var spinner = d3.selectAll(".spinner-border");
//     spinner.style("display", "none");
//     // 隐藏常规的热力图图例
//     d3.select("#thresholds").style("display", "none")
//     d3.select("#thresholds_diff").style("display", "block")


//     let contour_scale = heatMap_width / bins;

//     var heatMap_svg = d3.select("#center_heatMap_svg")
//         .attr("width", heatMap_width)
//         .attr("height", heatMap_height)


//     // 显示单选按钮选择的模型鲁棒性
//     var robustness1 = robustness_dict["M1"]
//     var robustness2 = robustness_dict["M2"]
//     var rob_expand = robustness1.map((val, index) => parseFloat((val - robustness2[index]).toFixed(4)))
//     //找到最大值和最小值
//     var min_value = 1e-5
//     var max_value = -1e5
//     for (var i = 0; i < rob_expand.length; i++) {
//         if (rob_expand[i] < min_value) {
//             min_value = rob_expand[i]
//         }
//         if (rob_expand[i] > max_value) {
//             max_value = rob_expand[i]
//         }
//     }
//     var thresholds_diff = [];
//     const dif_value = (max_value - min_value) / 5;
//     for (let i = 0; i < 6; i++) {
//         thresholds_diff.push(parseFloat((min_value + i * dif_value).toFixed(4)));
//     }
//     console.log("差异热力图中的thresholds_diff：", thresholds_diff)
//     // 标记颜色标尺
//     document.getElementById('threshold-A-value_diff').innerHTML = (thresholds_diff[0] * 1000).toFixed(1) + " ~ " + (thresholds_diff[1] * 1000).toFixed(1);
//     document.getElementById('threshold-B-value_diff').innerHTML = (thresholds_diff[1] * 1000).toFixed(1) + " ~ " + (thresholds_diff[2] * 1000).toFixed(1);
//     document.getElementById('threshold-C-value_diff').innerHTML = (thresholds_diff[2] * 1000).toFixed(1) + " ~ " + (thresholds_diff[3] * 1000).toFixed(1);
//     document.getElementById('threshold-D-value_diff').innerHTML = (thresholds_diff[3] * 1000).toFixed(1) + " ~ " + (thresholds_diff[4] * 1000).toFixed(1);
//     document.getElementById('threshold-E-value_diff').innerHTML = (thresholds_diff[4] * 1000).toFixed(1) + " ~ " + (thresholds_diff[5] * 1000).toFixed(1);
//     document.getElementById('threshold-F-value_diff').innerHTML = " ≥ " + (thresholds_diff[5] * 1000).toFixed(1);

//     // console.log("rob_expand: ", rob_expand)
//     // 计算等值线数据---------------------------------
//     let contours = d3.contours()
//         .size([bins, bins])
//         // .smooth(smooth)
//         .thresholds(thresholds_diff)
//         (rob_expand);
//     //颜色标尺
//     color = ["#005ab5", "#0080ff", "#84c1ff", "#fee5d9", "#fcbba1", "#fc9272"];
//     // 左侧热力图概览绘制------------------------------
//     selection_contour = heatMap_svg.select("#heatMap_g").selectAll("path")
//         .data(contours, d => d.value)
//         .join("path")
//         .attr("d", d3.geoPath())
//         .attr("fill", (d, i) => color[i])
//         .attr("stroke", "#fff")
//         .attr("stroke-width", "0.0001px")
//         .attr("transform", "translate(" + x + "," + y + "), scale(" + Number(length) / Number(bins) + ")") //要写在一起才行
//         .transition()
//         .duration(1000)
//         .attr("transform", "scale(" + contour_scale + ")") //只写scale，translate就默认是(0,0)

//     //绘制x，y轴---------------------------------------
//     // 绘制x轴
//     xAxis = d3.axisBottom(xScale)
//     gX = heatMap_svg.select("#x_g")
//         .call(xAxis);
//     // 绘制y轴
//     yAxis = d3.axisRight(yScale);
//     gY = heatMap_svg.select("#y_g")
//         .call(yAxis);
// }




/*
    改良热力图版本
*/
//将小于0的鲁棒性置0
function no_zero(robustness) {
    for (var i = 0; i < robustness.length; i++) {
        // 这里之所以要来回变换数据类型，是为了防止出错，导致热力图画不出来
        robustness[i] = parseFloat(robustness[i].toString()) >= 0 ? parseFloat(robustness[i].toString()) : 0
    }
    return robustness
}
// 根据阈值范围和类别修改图例
function change_legends(thresholds) {

    for (let i = 0; i < color.length; i++) {

        // 标记颜色范围
        if (i != color.length - 1) {
            // document.getElementById("threshold-" + (i).toString() + "-value").innerHTML = (thresholds[i] * 1000).toFixed(0) + " ~ " + (thresholds[i + 1] * 1000).toFixed(0);
            // document.getElementById("threshold-" + (i).toString() + "-value_right").innerHTML = (thresholds[i] * 1000).toFixed(0) + " ~ " + (thresholds[i + 1] * 1000).toFixed(0);
            document.getElementById("threshold-" + (i).toString() + "-value").innerHTML = (thresholds[i]).toFixed(0) + " ~ " + (thresholds[i+1]).toFixed(0);
            document.getElementById("threshold-" + (i).toString() + "-value_right").innerHTML = (thresholds[i]).toFixed(0) + " ~ " + (thresholds[i+1]).toFixed(0);
        } else {

                // document.getElementById("threshold-" + (i).toString() + "-value").innerHTML = " ≥ " + (thresholds[i] * 1000).toFixed(0);
                // document.getElementById("threshold-" + (i).toString() + "-value_right").innerHTML = " ≥ " + (thresholds[i] * 1000).toFixed(0);
                document.getElementById("threshold-" + (i).toString() + "-value").innerHTML = " ≥ " + (thresholds[i]).toFixed(0);
                document.getElementById("threshold-" + (i).toString() + "-value_right").innerHTML = " ≥ " + (thresholds[i]).toFixed(0)
        }
        // 修改图例颜色
        document.getElementById("threshold" + (i).toString() ).style.backgroundColor = color[i]
        document.getElementById("threshold" + (i).toString() + "_right" ).style.backgroundColor = color[i]

    }


}
// 修改热力图
function change_heatMap(is_transition=false,args=null) {
    // 判断当前是否要显示差异鲁棒性地图
    const radioDNNButtons = document.querySelectorAll('input[name="radio_model"]');
    // 获取当前被选中的单选按钮的索引
    var is_difference = false;
    function getCheckedIndex(radioElements) {
        for (let i = 0; i < radioElements.length; i++) {
            if (radioElements[i].checked) {
                return i;
            }
        }
        return -1;
    }
    var heatMapSelectedIndex = getCheckedIndex(radioDNNButtons);
    if (radioDNNButtons[heatMapSelectedIndex].value == "M1-M2"){
        is_difference = true
    }
    draw_heatMap_together(is_transition = is_transition, is_difference = is_difference, args = args)
}
// 这个绘制热力图包括，单个模型的鲁棒性分布，动态缩放裁剪的鲁棒性分布，以及鲁棒性差异
async function draw_heatMap_together(is_transition = false, is_difference = false, args = null) {
    /*
        bins是热力图‘分辨率’ = bins*bins
        transition表示是否需要过渡动画
        is_difference表示是否为
        args: 如果是动画必须传入args参数，这是一个字典，包括x, y, length
    */
    // console.log("is_difference", is_difference)
    // 停止转动加载图标
    var spinner = d3.selectAll(".spinner" +
        "" +
        "" +
        "-border");
    spinner.style("display", "none");

    // 记录缩放比例，配合d3.contours()使用
    var contour_scale = heatMap_width / bins;

    // 设置鲁棒性地图大小
    var heatMap_svg = d3.select("#center_heatMap_svg")
        .attr("width", heatMap_width)
        .attr("height", heatMap_height)
    var heatMap_svg_right = d3.select("#center_heatMap_svg_right")
        .attr("width", heatMap_width)
        .attr("height", heatMap_height)

    // 根据单选按钮，判断当前鲁棒性地图要展示的是单一模型的鲁棒性分布，还是两个鲁棒性之间的差异
    if (is_difference) { //如果是绘制差异
        var robustness1 = robustness_dict["M1"]
        var robustness2 = robustness_dict["M2"]
        var rob_expand = robustness1.map((val, index) => parseFloat((val - robustness2[index]).toFixed(4)))
        //找到最大值和最小值
        var min_value = 1e-5
        var max_value = -1e5
        for (var i = 0; i < rob_expand.length; i++) {
            if (rob_expand[i] < min_value) {
                min_value = rob_expand[i]
            }
            if (rob_expand[i] > max_value) {
                max_value = rob_expand[i]
            }
        }
        // 正数部分划分为3段
        const dif_value_positive = (max_value - 0) / 3;
        for(let i = 3; i < 6; i++){
            thresholds[i] = parseFloat((0 + (i-3) * dif_value_positive).toFixed(4));
        }
        // 负数部分也划分为3段
        const dif_value_negative = (min_value - 0) / 3;
        for(let i = 0; i < 3; i++){
            thresholds[i] = parseFloat((min_value - (i) * dif_value_negative).toFixed(4));
        }
        color = ["#005ab5", "#0080ff", "#84c1ff", "#fee5d9",
            "#fcbba1", "#fc9272"];
    } else {
        // 显示单选按钮选择的模型鲁棒性

        const selectedRadioModelButton = document.querySelector('input[name="radio_model"]:checked');
        const selectedValue = selectedRadioModelButton.value;//M1
        console.log("一个模型",selectedValue )

        // var robustness = robustness_dict[selectedValue]
        // var rob_expand = no_zero(robustness); //把0以下的鲁棒性值提高到0

        var confidence = confidence_dict[selectedValue]
        // var cof_expand = no_zero(confidence); //把0以下的鲁棒性值提高到0
        var cof_expand = confidence;
        var confidence_right = confidence_fore_dict[selectedValue]
        // var cof_right_expand = no_zero(confidence_right); //把0以下的鲁棒性值提高到0
        var cof_right_expand = confidence_right
         //找到最大值和最小值
        var number = cof_expand.map(parseFloat)
        var number2 = cof_right_expand.map(parseFloat)
        var min_value = Math.min(...number)
        var min_value2 = Math.min(...number2)
        // console.log(min_value2,min_value)
        if(min_value>min_value2){
            min_value=min_value2
        }




        // 修改阈值和图例
        var lowerBound = parseFloat(document.getElementById("lower_bound").value);

        var step = parseFloat(document.getElementById("threshold_step").value);
        console.log("lowerBound: ", lowerBound)
        console.log("step: ", step)
        // console.log(color.length)
        thresholds[0] = min_value

        for (let i = 0; i < color.length-1; i++) {
            let num = lowerBound + i * step;
            let str = num.toString();
            let dotIndex = str.indexOf('.');
            thresholds[i + 1] = parseFloat(dotIndex == -1 ? str : str.substring(0, dotIndex + 6)); //只保留五位小数
        }
        // color = ["#fee5d9", "#fcbba1", "#fc9272", "#fb6a4a", "#de2d26", "#a50f15"];
        // color = ['#fdd1c1', '#fcbba1', '#faa57f', '#f9845b', '#f7683c', '#f03b20', '#e31a1c', '#bd0026', '#800026'];
        // color = ['#fee5d9', '#fdd1c1', '#fcbba1', '#faa57f', '#f07e5e', '#d9583c', '#b5352b', '#8b1a1a', '#620f0f', '#3e0a0a']
        // color = ['#fee5d9', '#feedc3', '#ffe8ad', '#ffe298', '#ffc375', '#ffa551', '#ff853b', '#ff5e28', '#ff3d1f', '#ff2a1a']
        // color = ['#fee5d9', '#fdd1c1', '#fcbba1', '#faa57f', '#f07e5e', '#d9583c', '#b5352b', '#8b1a1a', '#620f0f', '#3e0a0a']
        //
        // color =  ['#fff5eb', '#fee6ce', '#fdd0a2', '#fdae6b', '#fd8d3c', '#f16913', '#d94801', '#a63603', '#7f2704']; //9-class Oranges
        // color = ['#fcfbfd', '#efedf5', '#dadaeb', '#bcbddc', '#9e9ac8', '#807dba', '#6a51a3', '#54278f', '#3f007d']; //9-class Purples
        // color = ['#f7fcf0', '#e0f3db', '#ccebc5', '#a8ddb5', '#7bccc4', '#4eb3d3', '#2b8cbe', '#0868ac', '#084081']; //9-class GnBu

        // // sequential 
        // // multi-hue
        // // 9-class BuGn
        // color = ['#f7fcfd','#e5f5f9','#ccece6','#99d8c9','' +
        // '#66c2a4','#41ae76','#238b45','#006d2c','#00441b']
        // // 9-class BuPu
        // color = ['#f7fcfd','#e0ecf4','#bfd3e6','#9ebcda',
        //     '#8c96c6','#8c6bb1','#88419d','#810f7c','#4d004b']
        // // 9-class GnBu
        // color = ['#f7fcf0','#e0f3db','#ccebc5','#a8ddb5',
        //     '#7bccc4','#4eb3d3','#2b8cbe','#0868ac','#084081']
        // // 9-class OrRd
        // color = ['#fff7ec','#fee8c8','#fdd49e','#fdbb84','#fc8d59',
        //     '#ef6548','#d7301f','#b30000','#7f0000']
        // // 9-class PuBu
        // color = ['#fff7fb','#ece7f2','#d0d1e6','#a6bddb','#74a9cf','#3690c0','#0570b0','#045a8d','#023858']
        // // 9-class PuBuGn
        // color = ['#fff7fb','#ece2f0','#d0d1e6','#a6bddb','#67a9cf','#3690c0','#02818a','#016c59','#014636']
        // // 9-class PuRd
        // color = ['#f7f4f9','#e7e1ef','#d4b9da','#c994c7','#df65b0','#e7298a','#ce1256','#980043','#67001f']
        // // 9-class RdPu
        // color = ['#fff7f3','#fde0dd','#fcc5c0','#fa9fb5','#f768a1','#dd3497','#ae017e','#7a0177','#49006a']
        // // 9-class YlGn
        // color = ['#ffffe5','#f7fcb9','#d9f0a3','#addd8e','#78c679','#41ab5d','#238443','#006837','#004529']
        // // 9-class YlGnBu
        // color = ['#ffffd9','#edf8b1','#c7e9b4','#7fcdbb','#41b6c4','#1d91c0','#225ea8','#253494','#081d58']
        // // 9-class YlOrBr
        // color = ['#ffffe5','#fff7bc','#fee391','#fec44f','#fe9929','#ec7014','#cc4c02','#993404','#662506']
        // // 9-class YlOrRd
        // color = ['#ffffcc','#ffeda0','#fed976','#feb24c','#fd8d3c','#fc4e2a','#e31a1c','#bd0026','#800026']
        // // single hue
        // // 9-class Blues
        // color = ['#f7fbff','#deebf7','#c6dbef','#9ecae1','#6baed6','#4292c6','#2171b5','#08519c','#08306b']
        // // 9-class Greens
        // color = ['#f7fcf5','#e5f5e0','#c7e9c0','#a1d99b','#74c476','#41ab5d','#238b45','#006d2c','#00441b']
        // // 9-class Greys
        // color = ['#ffffff','#f0f0f0','#d9d9d9','#bdbdbd','#969696','#737373','#525252','#252525','#000000']
        // // 9-class Oranges
        // color = ['#fff5eb','#fee6ce','#fdd0a2','#fdae6b','#fd8d3c','#f16913','#d94801','#a63603','#7f2704']
        // // 9-class Purples
        // color = ['#fcfbfd','#efedf5','#dadaeb','#bcbddc','#9e9ac8','#807dba','#6a51a3','#54278f','#3f007d']
        // // 9-class Reds
        // color = ['#fff5f0','#fee0d2','#fcbba1','#fc9272','#fb6a4a','#ef3b2c','#cb181d','#a50f15','#67000d']
        // // diverging
        // // 9-class BrBG
        // color = ['#8c510a','#bf812d','#dfc27d','#f6e8c3','#f5f5f5','#c7eae5','#80cdc1','#35978f','#01665e']
        // // color.reverse();
        // // 9-class PiYG
        // color = ['#c51b7d','#de77ae','#f1b6da','#fde0ef','#f7f7f7','#e6f5d0','#b8e186','#7fbc41','#4d9221']
        // color.reverse();
        // // 9-class PRGn
        // color = ['#762a83','#9970ab','#c2a5cf','#e7d4e8','#f7f7f7',
        //     '#d9f0d3','#a6dba0','#5aae61','#1b7837']
        // color.reverse();
        // // 9-class PuOr
        // color = ['#b35806','#e08214','#fdb863','#fee0b6','#f7f7f7',
        //     '#d8daeb','#b2abd2','#8073ac','#542788']
        // color.reverse();
        // // 9-class RdBu
        // color = ['#b2182b','#d6604d','#f4a582','#fddbc7',
        //     '#f7f7f7','#d1e5f0','#92c5de','#4393c3','#2166ac']
        // color.reverse();
        // 9-class RdYlBu
        // color = [
        //     '#d73027','#f46d43','#fdae61','#fee090',
        //     '#ffffbf',
        //     '#e0f3f8','#abd9e9','#74add1','#4575b4'
        // ]
        // color = ['#5281f3','#4a88f4','#4190f6','#3998f7',
        //     '#309ff9','#28a7fa','#1faffc',
        //     '#17b6fd','#0ebeff']
       // color = ['#4575b4','#5885bd','#6c95c5','#7fa4ce',
       //     '#93b4d6','#a6c4df','#b9d4e7',
       //     '#cde3f0','#e0f3f8']
       //  color = ['#4575b4','#6c95c5','#93b4d6','#b9d4e7',
       //     '#e0f3f8','#feefea','#e7c6bc','#eba993',
       //      '#f08b6b', '#f46d43']



        //  color = [
        //     '#f46d43','#fdae61','#fee090',
        //     '#abd9e9','#74add1'
        // ]
        // color = ['#4575b4', '#74add1', '#abd9e9', '#e0f3f8',
        //     // '#e8f6fa'
        //     ];
        // color.reverse();
        console.log(thresholds)
    }

    // 修改图例
    change_legends(thresholds) // threshold在main.html中定义
    var contours = d3.contours()
        .size([bins, bins])
        .thresholds(thresholds)
        .smooth(true)
        (cof_expand);
    var contours_right = d3.contours()
        .size([bins, bins])
        .thresholds(thresholds)
        .smooth(true)
        (cof_right_expand);
    // var contours = d3.contourDensity()
    //     .x((d, i) => Math.sqrt(robustness.length) - Math.floor(i / Math.sqrt(robustness.length)))
    //     .y((d, i) => i % Math.sqrt(robustness.length)) 
    //     .weight(d => d)
    //     .size([bins, bins])
    //     .bandwidth([0.6]) 
    //     .cellSize([1])
    //     .thresholds(thresholds)
    //     (rob_expand);
    // console.log(contours)
    console.log("cof_expand的形状", cof_expand.length)
    console.log("cof_expand的内容", cof_expand)
    console.log("cof_right_expand的形状", cof_right_expand.length)
    console.log("cof_right_expand的内容", cof_right_expand)

    // 判断是否用动画绘制热力图
    if (is_transition) {
        var x = args.x
        var y = args.y
        var length = args.length
        selection_contour = heatMap_svg.select("#heatMap_g").selectAll("path")
            .data(contours, d => d.value)
            .join("path")
            .attr("d", d3.geoPath())
            .attr("fill", (d, i) => color[i])
            .attr("stroke", "#fff")
            .attr("stroke-width", "0.0001px")
            .attr("transform", "translate(" + x + "," + y + "), scale(" + Number(length) / Number(bins) + ")") //要写在一起才行
            .transition()
            .duration(1000)
            .attr("transform", "scale(" + contour_scale + ")") //只写scale，translate就默认是(0,0)
        selection_contour_right = heatMap_svg_right.select("#heatMap_g_right").selectAll("path")
            .data(contours_right, d => d.value)
            .join("path")
            .attr("d", d3.geoPath())
            .attr("fill", (d, i) => color[i])
            .attr("stroke", "#fff")
            .attr("stroke-width", "0.0001px")
            .attr("transform", "translate(" + x + "," + y + "), scale(" + Number(length) / Number(bins) + ")") //要写在一起才行
            .transition()
            .duration(1000)
            .attr("transform", "scale(" + contour_scale + ")") //只写scale，translate就默认是(0,0)
    } else {
        selection_contour = heatMap_svg.select("#heatMap_g").selectAll("path")
            .data(contours, d => d.value)
            .join("path")
            .attr("d", d3.geoPath())
            .attr("transform", "scale(" + contour_scale + ")")
            .attr("fill", (d, i) => color[i])
            // .attr("stroke", "#fff")
            // .attr("stroke-width", "0.0001px")
        selection_contour_right = heatMap_svg_right.select("#heatMap_g_right").selectAll("path")
            .data(contours_right, d => d.value)
            .join("path")
            .attr("d", d3.geoPath())
            .attr("transform", "scale(" + contour_scale + ")")
            .attr("fill", (d, i) => color[i])
            // .attr("stroke", "#fff")
            // .attr("stroke-width", "0.0001px")
    }
    //绘制x，y轴---------------------------------------
    // 绘制x轴
    xAxis = d3.axisBottom(xScale)
    gX = heatMap_svg.select("#x_g")
        .call(xAxis);
    gX_right = heatMap_svg_right.select("#x_g_right")
        .call(xAxis);
    // 绘制y轴
    yAxis = d3.axisRight(yScale);
    gY = heatMap_svg.select("#y_g")
        .call(yAxis);
    gY_right = heatMap_svg_right.select("#y_g_right")
        .call(yAxis);
}