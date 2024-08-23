/*
    这个js主要是用来控制一些小控件
*/
var heatMap_svg = d3.select("#center_heatMap_svg");
var imageGrid_svg = d3.select("#imageGrid_svg");
var imageTypeGrid_svg = d3.select("#imageTypeGrid_svg");
var heatMap_svg_right = d3.select("#center_heatMap_svg_right");
var imageGrid_svg_right = d3.select("#imageGrid_svg_right");
var imageTypeGrid_svg_right = d3.select("#imageTypeGrid_svg_right");
var img_number = 0;

// 分辨率事件~~~~~~~~~~~~~~~~~~~~~~~~~~~
d3.select("#resolution").on("change", () => {
    // 先清除东西
    const clearBtn = document.querySelector("#btn_clear");
    clearBtn.click()
    bins = document.getElementById("resolution").value; //bing全局变量在main.html中定义
    //获取鲁棒性(鲁棒性等信息保存在全局变量中)，之后再调用其他方法
    get_information_from_python(extent = extent, xScale = xScale, yScale = yScale).then(function () {
        zoom_updata_all() //该函数来源于zoom.js
    })
});
// 反距离指数事件~~~~~~~~~~~~~~~~~~~~~~~~~~~
d3.select("#idw_p").on("change", () => {
    // 先清除东西
    const clearBtn = document.querySelector("#btn_clear");
    clearBtn.click()
    bins = document.getElementById("resolution").value; //bing全局变量在main.html中定义
    //获取鲁棒性(鲁棒性等信息保存在全局变量中)，之后再调用其他方法
    get_information_from_python(extent = extent, xScale = xScale, yScale = yScale).then(function () {
        zoom_updata_all() //该函数来源于zoom.js
    })
});


// 设置阈值事件~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// const setThresholdBtn = document.querySelector("#threshold_save")
// setThresholdBtn.addEventListener("click", function () {
//     change_heatMap();
// })
d3.select("#lower_bound").on("change", () => {
        change_heatMap();


});
d3.select("#threshold_step").on("change", () => {
        change_heatMap();
});


// 色卡更换

const color1Btn = document.querySelector("#radio_color1");
const color2Btn = document.querySelector("#radio_color2");
const color3Btn = document.querySelector("#radio_color3");
color1Btn.addEventListener("click", function () {
    color = ['#4575b4','#6c95c5','#93b4d6','#b9d4e7',
                   '#e0f3f8','#feefea','#e7c6bc','#eba993',
                    '#f08b6b', '#f46d43']
    color.reverse()

    change_heatMap();
})
color2Btn.addEventListener("click", function () {
    color = ['#b1e8a5','#c6edbe','#d6f2d0','#e7f7e3',
           '#f7fcf6','#fff6f5','#fee3e1','#fed1cd',
            '#fdbeb9', '#f89f99']
    color.reverse()

    change_heatMap();

})
color3Btn.addEventListener("click", function () {
    color = ['#f3e619','#c9e019','#8bd543','#4cc469',
        '#1fa884','#1c928c','#237d8e','#2f5f8d',
        '#3a4a8a','#44247a']
    color.reverse()

    change_heatMap();

})