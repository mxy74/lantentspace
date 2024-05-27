/*
    这个js主要用来控制缩放事件
    这里的xScale和yScale来自main.html中的全局变量
*/

//统一修改
function zoom_updata_all() {
    //先获取当前选择鲁棒性地图模型
    const radioDNNButtons = document.querySelectorAll('input[name="radio_model"]');
    // 先储存原本的选择
    // 获取当前被选中的单选按钮的索引
    function getCheckedIndex(radioElements) {
        for (let i = 0; i < radioElements.length; i++) {
            if (radioElements[i].checked) {
                return i;
            }
        }
        return -1;
    }
    var originalSelectedIndex = getCheckedIndex(radioDNNButtons);
    if (radioDNNButtons[originalSelectedIndex].value != "M1-M2") {
        //修改类别网格 (函数来自于image_type_grid.js)，只有研究单个模型的时候才要显示类别网格
        change_imageTypeGrid()
    }
    //修改热力图（函数来自heat_map.js）
    change_heatMap()
    //修改图片网格 (函数来自于image_grid.js)
    change_imageGrid()

}

// 修改单击事件，以及划线比较事件中产生的点，线，图的位置
// 这里面的对象都保存了他们对应真实二维潜空间的坐标，依据这个坐标进行还原
function zoom_the_click_result(new_xScale, new_yScale) {
    // 单机事件产生的结果·································································
    // 修改点（添加偏移）
    heatMap_svg.select("#imageAndCircle_g").selectAll("circle")
        .attr("cx", function () {
            var pos_x = parseFloat(d3.select(this).attr("pos_x"))
            var new_cx = offset_xScale(new_xScale, pos_x)
            return new_cx;
        }).attr("cy", function () {
            var pos_y = parseFloat(d3.select(this).attr("pos_y"));
            var new_cy = offset_yScale(new_yScale, pos_y)
            return new_cy;
        })
    heatMap_svg_right.select("#imageAndCircle_g_right").selectAll("circle")
        .attr("cx", function () {
            var pos_x = parseFloat(d3.select(this).attr("pos_x"))
            var new_cx = offset_xScale(new_xScale, pos_x)
            return new_cx;
        }).attr("cy", function () {
            var pos_y = parseFloat(d3.select(this).attr("pos_y"));
            var new_cy = offset_yScale(new_yScale, pos_y)
            return new_cy;
        })
    // 修改图片（添加偏移）
    heatMap_svg.select("#imageAndCircle_g").selectAll("image")
        .attr("x", function () {
            var pos_x = parseFloat(d3.select(this).attr("pos_x"))
            var new_cx = offset_xScale(new_xScale, pos_x);
            return new_cx;
        }).attr("y", function () {
            var pos_y = parseFloat(d3.select(this).attr("pos_y"));
            var new_cy = offset_yScale(new_yScale, pos_y);
            return new_cy;
        })
    heatMap_svg_right.select("#imageAndCircle_g_right").selectAll("image")
        .attr("x", function () {
            var pos_x = parseFloat(d3.select(this).attr("pos_x"))
            var new_cx = offset_xScale(new_xScale, pos_x);
            return new_cx;
        }).attr("y", function () {
            var pos_y = parseFloat(d3.select(this).attr("pos_y"));
            var new_cy = offset_yScale(new_yScale, pos_y);
            return new_cy;
        })
    //···················································································
    // 划线对比产生的结果
    // 修改线
    heatMap_svg.select("#imageCompare_g").selectAll("line")
        .attr("x1", function () {
            var pos_x1 = parseFloat(d3.select(this).attr("pos_x1"));
            var new_x1 = offset_xScale(new_xScale, pos_x1);
            return new_x1;
        })
        .attr("y1", function () {
            var pos_y1 = parseFloat(d3.select(this).attr("pos_y1"));
            var new_y1 = offset_yScale(new_yScale, pos_y1);
            return new_y1;
        })
        .attr("x2", function () {
            var pos_x2 = parseFloat(d3.select(this).attr("pos_x2"));
            var new_x2 = offset_xScale(new_xScale, pos_x2);
            return new_x2;
        })
        .attr("y2", function () {
            var pos_y2 = parseFloat(d3.select(this).attr("pos_y2"));
            var new_y2 = offset_yScale(new_yScale, pos_y2);
            return new_y2;
        })
    // 修改点
    heatMap_svg.select("#imageCompare_g").selectAll("circle")
        .attr("cx", function () {
            var pos_x = parseFloat(d3.select(this).attr("pos_x"))
            var new_cx = offset_xScale(new_xScale, pos_x)
            return new_cx;
        }).attr("cy", function () {
            var pos_y = parseFloat(d3.select(this).attr("pos_y"));
            var new_cy = offset_yScale(new_yScale, pos_y)
            return new_cy;
        })
    // 修改图例
    heatMap_svg.select("#imageCompare_g").selectAll("g").selectAll("path")
        .attr("transform", function () {
            var pos_x = parseFloat(d3.select(this).attr("pos_x"))
            var new_x = offset_xScale(new_xScale, pos_x)
            var pos_y = parseFloat(d3.select(this).attr("pos_y"))
            var new_y = offset_yScale(new_yScale, pos_y)
            // 这个icon_size在control.js中定义了
            return "translate(" + String(Number(new_x) - icon_size / 2) + "," + String(Number(new_y) - icon_size) + ")"
        })
}


//缩放-----------------------------------------
var new_xScale;
var new_yScale;
var heatMap_svg = d3.select("#center_heatMap_svg");
var imageGrid_svg = d3.select("#imageGrid_svg");
var imageTypeGrid_svg = d3.select("#imageTypeGrid_svg");
var heatMap_svg_right = d3.select("#center_heatMap_svg_right");
var imageGrid_svg_right = d3.select("#imageGrid_svg_right");
var imageTypeGrid_svg_right = d3.select("#imageTypeGrid_svg_right");
// 这里定义的zoom还没有绑定，只是纯纯定义了一个方法，具体绑定在control.js里面通过call方式绑定
zoom = d3.zoom()
    .on("zoom", function (event) {
        //修改三个主要的视图
        heatMap_svg.select("#heatMap_g").attr("transform", event.transform)
        imageGrid_svg.select("#imageGrid_g").attr("transform", event.transform)
        imageTypeGrid_svg.select("#imageTypeGrid_g").attr("transform", event.transform)
        //修改热力图中点击产生的圈和图片，以及划线对比产生的点线，和图例
        heatMap_svg.select("#imageAndCircle_g").attr("transform", event.transform)
        heatMap_svg.select("#imageCompare_g").attr("transform", event.transform)

        // mix
        heatMap_svg_right.select("#heatMap_g_right").attr("transform", event.transform)
        imageGrid_svg_right.select("#imageGrid_g_right").attr("transform", event.transform)
        imageTypeGrid_svg_right.select("#imageTypeGrid_g_right").attr("transform", event.transform)
        //修改热力图中点击产生的圈和图片，以及划线对比产生的点线，和图例
        heatMap_svg_right.select("#imageAndCircle_g_right").attr("transform", event.transform)
        heatMap_svg_right.select("#imageCompare_g_right").attr("transform", event.transform)

        //修改坐标轴
        new_xScale = event.transform.rescaleX(xScale)
        new_yScale = event.transform.rescaleY(yScale)
        gX.call(xAxis.scale(new_xScale));
        gY.call(yAxis.scale(new_yScale));
        domainX = new_xScale.domain();
        domainY = new_yScale.domain();
        extent.start_x = domainX[0];
        extent.end_x = domainX[1];
        extent.start_y = domainY[0];
        extent.end_y = domainY[1];
    })
    .on("end", async function (event) {
        var bins = document.getElementById("resolution").value;
        xScale = new_xScale;
        yScale = new_yScale;
        event.transform.x = 0;
        event.transform.y = 0;
        event.transform.k = 1;
        console.log("平移后的extent：",extent)
        //获取鲁棒性(鲁棒性等信息保存在全局变量中)，之后再调用其他方法
        get_information_from_python(extent = extent, bins = bins, xScale = xScale, yScale = yScale).then(function () {
            //让位移归为
            heatMap_svg.select("#heatMap_g").attr("transform", event.transform)
            heatMap_svg.select("#imageAndCircle_g").attr("transform", event.transform)
            imageGrid_svg.select("#imageGrid_g").attr("transform", event.transform)
            // 单击事件在热力图上产生的结果
            imageTypeGrid_svg.select("#imageTypeGrid_g").attr("transform", event.transform)
            heatMap_svg.select("#imageCompare_g").attr("transform", event.transform)
             // mix
            heatMap_svg_right.select("#heatMap_g_right").attr("transform", event.transform)
            heatMap_svg_right.select("#imageAndCircle_g_right").attr("transform", event.transform)
            imageGrid_svg_right.select("#imageGrid_g_right").attr("transform", event.transform)
            // 单击事件在热力图上产生的结果
            imageTypeGrid_svg_right.select("#imageTypeGrid_g_right").attr("transform", event.transform)
            heatMap_svg_right.select("#imageCompare_g_right").attr("transform", event.transform)
            // 让点击产生的结果保持不变
            zoom_the_click_result(new_xScale, new_yScale);
            //更新所有概览
            zoom_updata_all()
        })

    })




