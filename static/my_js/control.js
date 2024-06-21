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

//统一修改(有过度动画，主要是针对裁剪做的)
function control_updata_all_transition(x, y, length) {
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
    if (radioDNNButtons[originalSelectedIndex].value !== "M1-M2") {
        //修改类别网格 (函数来自于image_type_grid.js)
        change_imageTypeGrid()
    }
    var args = {
        "x": x,
        "y": y,
        "length": length
    }
    change_heatMap(true, args)
    //修改图片网格 (函数来自于image_grid.js)
    change_imageGrid()
    change_matrix()

}

// 导航栏工具栏-----------------------------------------------------------------------------------
// 数据集选择事件
const datasetsSelect = document.querySelector("#select_dataset_type_selection");
datasetsSelect.addEventListener("change", async function () {
    var dataset_type = document.getElementById("select_dataset_type_selection").value //在Document.querySelector()通过id获取才需要加#
    var modelSelect = document.querySelector("#select_model_selection");
    if (dataset_type === "CIFAR10") {
        // 清空原有选项
        modelSelect.innerHTML = "";
        // 添加新的选项
        var options = ["ResNet20", "ResNet32", "ResNet44", "ResNet56", "VGG11_BN", "VGG13_BN", "VGG16_BN", "VGG19_BN", "MobileNetV2_x0_5", "MobileNetV2_x0_75", "MobileNetV2_x1_0", "MobileNetV2_x1_4"];
        for (var i = 0; i < options.length; i++) {
            var option = document.createElement("option");
            option.text = options[i];
            modelSelect.add(option);
        }
        // document.querySelector("#task_category").innerHTML = "Category";

        document.querySelector("#model_title_name").innerHTML = document.querySelector("#select_model_selection").value;
    } else if (dataset_type === "GTSRB") {
        // 清空原有选项
        modelSelect.innerHTML = "";
        // 添加新的选项
        var options = ["ResNet18", "ResNet32", "ResNet44", "ResNet56", "VGG11_BN", "VGG13_BN", "VGG16_BN", "VGG19_BN", "MobileNetV2_x0_5", "MobileNetV2_x0_75", "MobileNetV2_x1_0", "MobileNetV2_x1_4"];
        for (var i = 0; i < options.length; i++) {
            var option = document.createElement("option");
            option.text = options[i];
            modelSelect.add(option);
        }
        // document.querySelector("#task_category").innerHTML = "Category";

        document.querySelector("#model_title_name").innerHTML = document.querySelector("#select_model_selection").value;
    }else if (dataset_type == "SteeringAngle") {
        // 清空原有选项
        modelSelect.innerHTML = "";
        // 添加新的选项
        var options = ["ResNet34_regre", "ResNet50_regre", "ResNet101_regre"];
        for (var i = 0; i < options.length; i++) {
            var option = document.createElement("option");
            option.text = options[i];
            modelSelect.add(option);
        }
        document.querySelector("#task_category").innerHTML = "Angle";

        document.querySelector("#model_title_name").innerHTML = document.querySelector("#select_model_selection").value;
    }
    await prepare_shared_data(dataset_type)
    var model_id = "M1"
    var model_name = document.getElementById("select_model_selection").value//在Document.querySelector()通过id获取才需要加#
    await prepare_DNN_data(model_id, model_name);
    //获取鲁棒性(鲁棒性等信息保存在全局变量中)
    get_information_from_python(extent = extent, bins = bins, xScale = xScale, yScale = yScale).then(function () {
        //绘制热力图
        change_heatMap()
        //绘制图片概览
        change_imageGrid()
        //绘制类别概览
        change_imageTypeGrid()
        change_matrix()
    })
})

// 模型选择事件(选择完成后，后端会先加载相应的模型，并获取鲁棒性相关的数据)~~~~~~~~~~~~~~~~~~~~~~~~~
// 第一个模型
const modelSelect1 = document.querySelector("#select_model_selection");
modelSelect1.addEventListener("change", async function () {
    // 修改模型后，需要清除上面的图片
    const clearBtn = document.querySelector("#btn_clear");
    clearBtn.click()
    var model_id = "M1";
    var model_name = document.getElementById("select_model_selection").value//在Document.querySelector()通过id获取才需要加#
    document.getElementById("model_title_name").innerHTML = model_name
    await prepare_DNN_data(model_id, model_name);
    //获取鲁棒性(鲁棒性等信息保存在全局变量中)
    get_information_from_python(extent = extent, bins = bins, xScale = xScale, yScale = yScale).then(function () {
        //绘制热力图
        change_heatMap()
        //绘制类别概览
        change_imageTypeGrid()
        change_matrix()
    })
})
// 第二个模型
const modelSelect2 = document.querySelector("#select_model_selection_compare");
modelSelect2.addEventListener("change", async function () {
    // 修改模型后，需要清除上面的图片
    const clearBtn = document.querySelector("#btn_clear");
    clearBtn.click()
    var model_id = "M2";
    var model_name = document.getElementById("select_model_selection_compare").value //在Document.querySelector()通过id获取才需要加#
    await prepare_DNN_data(model_id, model_name);
    //获取鲁棒性(鲁棒性等信息保存在全局变量中)
    get_information_from_python(extent = extent, xScale = xScale, yScale = yScale).then(function () {
        //绘制热力图
        change_heatMap()
        //绘制类别概览
        change_imageTypeGrid()
        change_matrix()
    })
})
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


// 缩小事件~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
d3.select("#zoom-in").on("click", () => { heatMap_svg.call(zoom.scaleBy, 1.3); });

// 放大事件~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
d3.select("#zoom-out").on("click", () => { heatMap_svg.call(zoom.scaleBy, 0.6); });

// 选框裁剪缩放事件~~~~~~~~~~~~~~~~~~~~~~
const cropBtn = document.querySelector("#btn_crop");
var crop_start_pos, pos, length_x, length_y, square_length;
let crop_all = d3.brush()
    .on("brush", function (event) {
        crop_start_pos = event.selection[0]
        pos = event.selection[1]
        length_x = Math.abs(pos[0] - crop_start_pos[0]);
        length_y = Math.abs(pos[1] - crop_start_pos[1]);
        square_length = Math.min(length_x, length_y); //取最短，目的是为了保持方形  
        // 热力图加选框
        heatMap_svg.select("#selection_inner_rect")
            .attr("x", function () {
                if (crop_start_pos[0] < pos[0]) {
                    return crop_start_pos[0];
                } else {
                    return crop_start_pos[0] - square_length;
                }
            })
            .attr("y", function () {
                if (crop_start_pos[1] < pos[1]) {
                    return crop_start_pos[1];
                } else {
                    return crop_start_pos[1] - square_length;
                }
            })
            .attr("height", square_length)
            .attr("width", square_length)
            .attr("fill-opacity", 0.5);

        imageGrid_svg.select("#selection_inner_rect")
            .attr("x", function () {
                if (crop_start_pos[0] < pos[0]) {
                    return crop_start_pos[0];
                } else {
                    return crop_start_pos[0] - square_length;
                }
            })
            .attr("y", function () {
                if (crop_start_pos[1] < pos[1]) {
                    return crop_start_pos[1];
                } else {
                    return crop_start_pos[1] - square_length;
                }
            })
            .attr("height", square_length)
            .attr("width", square_length)
            .attr("fill-opacity", 0.5);

        imageTypeGrid_svg.select("#selection_inner_rect")
            .attr("x", function () {
                if (crop_start_pos[0] < pos[0]) {
                    return crop_start_pos[0];
                } else {
                    return crop_start_pos[0] - square_length;
                }
            })
            .attr("y", function () {
                if (crop_start_pos[1] < pos[1]) {
                    return crop_start_pos[1];
                } else {
                    return crop_start_pos[1] - square_length;
                }
            })
            .attr("height", square_length)
            .attr("width", square_length)
            .attr("fill-opacity", 0.5);

        // 热力图加选框
        heatMap_svg_right.select("#selection_inner_rect_right")
            .attr("x", function () {
                if (crop_start_pos[0] < pos[0]) {
                    return crop_start_pos[0];
                } else {
                    return crop_start_pos[0] - square_length;
                }
            })
            .attr("y", function () {
                if (crop_start_pos[1] < pos[1]) {
                    return crop_start_pos[1];
                } else {
                    return crop_start_pos[1] - square_length;
                }
            })
            .attr("height", square_length)
            .attr("width", square_length)
            .attr("fill-opacity", 0.5);

        imageGrid_svg_right.select("#selection_inner_rect_right")
            .attr("x", function () {
                if (crop_start_pos[0] < pos[0]) {
                    return crop_start_pos[0];
                } else {
                    return crop_start_pos[0] - square_length;
                }
            })
            .attr("y", function () {
                if (crop_start_pos[1] < pos[1]) {
                    return crop_start_pos[1];
                } else {
                    return crop_start_pos[1] - square_length;
                }
            })
            .attr("height", square_length)
            .attr("width", square_length)
            .attr("fill-opacity", 0.5);

        imageTypeGrid_svg_right.select("#selection_inner_rect_right")
            .attr("x", function () {
                if (crop_start_pos[0] < pos[0]) {
                    return crop_start_pos[0];
                } else {
                    return crop_start_pos[0] - square_length;
                }
            })
            .attr("y", function () {
                if (crop_start_pos[1] < pos[1]) {
                    return crop_start_pos[1];
                } else {
                    return crop_start_pos[1] - square_length;
                }
            })
            .attr("height", square_length)
            .attr("width", square_length)
            .attr("fill-opacity", 0.5);
    })
    .on("end", function () {
        let node_inner = document.querySelector("#center_heatMap_svg").querySelector("#selection_inner_rect");
        var bins = document.getElementById("resolution").value;

        // 隐藏那个滑选选框，因为它在默认情况下就是display：none，所以在下次再画的时候它会自动解除
        // 热力图隐藏crop
        if (document.getElementById("heatmap_crop_g") != null) {
            heatMap_svg.select("#heatmap_crop_g .selection").style("display", "none")
        }
        if (document.getElementById("heatmap_crop_g_right") != null) {
            heatMap_svg_right.select("#heatmap_crop_g_right .selection").style("display", "none")
        }
        //图片概览隐藏crop
        if (document.getElementById("imageGrid_crop_g_right") != null) {
            imageGrid_svg_right.select("#imageGrid_crop_g_right .selection").style("display", "none")
        }
        if (document.getElementById("imageGrid_crop_g") != null) {
            imageGrid_svg.select("#imageGrid_crop_g .selection").style("display", "none")
        }
        //图片类别概览隐藏crop
        if (document.getElementById("imageTypeGrid_crop_g") != null) {
            imageTypeGrid_svg.select("#imageTypeGrid_crop_g .selection").style("display", "none")
        }
         if (document.getElementById("imageTypeGrid_crop_g_right") != null) {
            imageTypeGrid_svg_right.select("#imageTypeGrid_crop_g_right .selection").style("display", "none")
        }


        //修改坐标轴
        extent.start_x = xScale.invert(node_inner.getAttribute("x"));
        extent.end_x = xScale.invert(Number(node_inner.getAttribute("x")) + Number(node_inner.getAttribute("width")));
        extent.start_y = yScale.invert(node_inner.getAttribute("y"));
        extent.end_y = yScale.invert(Number(node_inner.getAttribute("y")) + Number(node_inner.getAttribute("height")));
        // 定义X轴scale
        xScale = d3.scaleLinear()
            .domain([extent.start_x, extent.end_x])
            .range([0, heatMap_width]);
        // 定义Y轴scale
        yScale = d3.scaleLinear()
            .domain([extent.start_y, extent.end_y])
            .range([0, heatMap_height]);
        console.log("缩放裁剪后的extent：", extent)
        //获取鲁棒性(鲁棒性等信息保存在全局变量中)，之后再调用其他方法
        get_information_from_python(extent = extent, bins = bins, xScale = xScale, yScale = yScale).then(function () {
            //length记录一下长度
            var length = node_inner.getAttribute("width")
            // 隐藏那个选框
            heatMap_svg.select("#selection_inner_rect")
                .attr("height", 0)
                .attr("width", 0)
            imageGrid_svg.select("#selection_inner_rect")
                .attr("height", 0)
                .attr("width", 0)
            imageTypeGrid_svg.select("#selection_inner_rect")
                .attr("height", 0)
                .attr("width", 0)
             // 隐藏那个选框
            heatMap_svg_right.select("#selection_inner_rect_right")
                .attr("height", 0)
                .attr("width", 0)
            imageGrid_svg_right.select("#selection_inner_rect_right")
                .attr("height", 0)
                .attr("width", 0)
            imageTypeGrid_svg_right.select("#selection_inner_rect_right")
                .attr("height", 0)
                .attr("width", 0)
            //更新坐标
            gX.call(xAxis.scale(xScale));
            gY.call(yAxis.scale(yScale));
            gX_right.call(xAxis.scale(xScale));
            gY_right.call(yAxis.scale(yScale));
            //更新所有概览
            control_updata_all_transition(x = node_inner.getAttribute("x"), y = node_inner.getAttribute("y"), length = length)
        })
    });
// 缩放裁剪
cropBtn.addEventListener("click", function () {
    if (cropBtn.checked) {
        // 默认点击清楚按钮，将图片给清除掉
        clearBtn.click();
        // 关闭brushBtn按钮
        if (brushBtn.checked == true) {
            brushBtn.click();
        }
        // 关闭panBtn
        if (panBtn.checked == true) {
            panBtn.click();
        }
        // 关闭locateBtn
        if (locateBtn.checked == true) {
            locateBtn.click();
        }
        // 关闭lineBtn
        if (lineBtn.checked == true) {
            lineBtn.click();
        }
        // 热力图概览添加crop
        if (document.getElementById("heatmap_crop_g") === null) {
            heatMap_svg.append("g")
                .attr("class", "brush")
                .attr("id", "heatmap_crop_g")
                .call(crop_all);
        }
        if (document.getElementById("heatmap_crop_g_right") === null) {
            heatMap_svg_right.append("g")
                .attr("class", "brush")
                .attr("id", "heatmap_crop_g_right")
                .call(crop_all);
        }
        //图片概览添加crop
        if (document.getElementById("imageGrid_crop_g") === null) {
            imageGrid_svg.append("g")
                .attr("class", "brush")
                .attr("id", "imageGrid_crop_g")
                .call(crop_all);
        }
        if (document.getElementById("imageGrid_crop_g_right") === null) {
            imageGrid_svg_right.append("g")
                .attr("class", "brush")
                .attr("id", "imageGrid_crop_g_right")
                .call(crop_all);
        }
        //图片类别概览添加crop
        if (document.getElementById("imageTypeGrid_crop_g") === null) {
            imageTypeGrid_svg.append("g")
                .attr("class", "brush")
                .attr("id", "imageTypeGrid_crop_g")
                .call(crop_all);
        }
        if (document.getElementById("imageTypeGrid_crop_g_right") === null) {
            imageTypeGrid_svg_right.append("g")
                .attr("class", "brush")
                .attr("id", "imageTypeGrid_crop_g_right")
                .call(crop_all);
        }
    } else {
        // 删除crop
        if (document.getElementById("heatmap_crop_g") !== null) {
            document.getElementById("heatmap_crop_g").remove();
        }
        if (document.getElementById("imageGrid_crop_g") !== null) {
            document.getElementById("imageGrid_crop_g").remove();
        }
        if (document.getElementById("imageTypeGrid_crop_g") !== null) {
            document.getElementById("imageTypeGrid_crop_g").remove();
        }
        if (document.getElementById("heatmap_crop_g_right") !== null) {
            document.getElementById("heatmap_crop_g_right").remove();
        }
        if (document.getElementById("imageGrid_crop_g_right") !== null) {
            document.getElementById("imageGrid_crop_g_right").remove();
        }
        if (document.getElementById("imageTypeGrid_crop_g_right") !== null) {
            document.getElementById("imageTypeGrid_crop_g_right").remove();
        }
        // 隐藏那个选框
        heatMap_svg.select("#selection_inner_rect")
            .attr("height", 0)
            .attr("width", 0)
        imageGrid_svg.select("#selection_inner_rect")
            .attr("height", 0)
            .attr("width", 0)
        imageTypeGrid_svg.select("#selection_inner_rect")
            .attr("height", 0)
            .attr("width", 0)
         // 隐藏那个选框
        heatMap_svg_right.select("#selection_inner_rect_right")
            .attr("height", 0)
            .attr("width", 0)
        imageGrid_svg_right.select("#selection_inner_rect_right")
            .attr("height", 0)
            .attr("width", 0)
        imageTypeGrid_svg_right.select("#selection_inner_rect_right")
            .attr("height", 0)
            .attr("width", 0)
    }
})
// 平移事件~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
const panBtn = document.querySelector("#btn_pan");
panBtn.addEventListener("click", function () {
    if (panBtn.checked) {
        // 关闭cropBtn
        if (cropBtn.checked == true) {
            cropBtn.click();
        }
        // 关闭brushBtn
        if (brushBtn.checked == true) {
            brushBtn.click();
        }
        // 关闭locateBtn
        if (locateBtn.checked == true) {
            locateBtn.click();
        }
        // 关闭lineBtn
        if (lineBtn.checked == true) {
            lineBtn.click();
        }
        heatMap_svg.style("cursor", "move");
        imageGrid_svg.style("cursor", "move");
        imageTypeGrid_svg.style("cursor", "move");
        heatMap_svg_right.style("cursor", "move");
        imageGrid_svg_right.style("cursor", "move");
        imageTypeGrid_svg_right.style("cursor", "move");
        // 执行zoom
        heatMap_svg.call(zoom)
            .on("dblclick.zoom", null); //禁用双击缩放
        imageGrid_svg.call(zoom)
            .on("dblclick.zoom", null); //禁用双击缩放
        imageTypeGrid_svg.call(zoom)
            .on("dblclick.zoom", null); //禁用双击缩放
        heatMap_svg_right.call(zoom)
            .on("dblclick.zoom", null); //禁用双击缩放
        imageGrid_svg_right.call(zoom)
            .on("dblclick.zoom", null); //禁用双击缩放
        imageTypeGrid_svg_right.call(zoom)
            .on("dblclick.zoom", null); //禁用双击缩放
    } else {
        heatMap_svg.style("cursor", "default");
        imageGrid_svg.style("cursor", "default");
        imageTypeGrid_svg.style("cursor", "default");
        heatMap_svg_right.style("cursor", "default");
        imageGrid_svg_right.style("cursor", "default");
        imageTypeGrid_svg_right.style("cursor", "default");
        heatMap_svg.on(".zoom", null);//禁用zoom
        imageGrid_svg.on(".zoom", null);//禁用zoom
        imageTypeGrid_svg.on(".zoom", null);//禁用zoom
        heatMap_svg_right.on(".zoom", null);//禁用zoom
        imageGrid_svg_right.on(".zoom", null);//禁用zoom
        imageTypeGrid_svg_right.on(".zoom", null);//禁用zoom
    }
})

// 刷选框对应事件~~~~~~~~~~~~~~~~~~~~~~~
const brushBtn = document.querySelector("#btn_brush");
// map brush功能(必须要分别创建才不会卡，可能是不断递归导致的结果，所以需要创建不同的brush)
let heatMap_brush = d3.brush()
    .on("start brush", function (event) { //不加start会有一个小bug（就是选框后，点击空白，只有一个区域的选框消失，其他的没有消失）
        heatMap_svg_right.select("#heatmap_brush_g_right").call(d3.brush().move, event.selection)
        imageGrid_svg.select("#imageGrid_brush_g").call(d3.brush().move, event.selection)
        imageTypeGrid_svg.select("#imageTypeGrid_brush_g").call(d3.brush().move, event.selection)
        imageGrid_svg_right.select("#imageGrid_brush_g_right").call(d3.brush().move, event.selection)
        imageTypeGrid_svg_right.select("#imageTypeGrid_brush_g_right").call(d3.brush().move, event.selection)
    })
let heatMap_brush_right = d3.brush()
    .on("start brush", function (event) { //不加start会有一个小bug（就是选框后，点击空白，只有一个区域的选框消失，其他的没有消失）
        heatMap_svg.select("#heatmap_brush_g").call(d3.brush().move, event.selection)
        imageGrid_svg.select("#imageGrid_brush_g").call(d3.brush().move, event.selection)
        imageTypeGrid_svg.select("#imageTypeGrid_brush_g").call(d3.brush().move, event.selection)
        imageGrid_svg_right.select("#imageGrid_brush_g_right").call(d3.brush().move, event.selection)
        imageTypeGrid_svg_right.select("#imageTypeGrid_brush_g_right").call(d3.brush().move, event.selection)
    })
let imageGrid_brush_right = d3.brush()
    .on("start brush", function (event) {
        heatMap_svg.select("#heatmap_brush_g").call(d3.brush().move, event.selection)
        heatMap_svg_right.select("#heatmap_brush_g_right").call(d3.brush().move, event.selection)
        imageTypeGrid_svg.select("#imageTypeGrid_brush_g").call(d3.brush().move, event.selection)
        imageGrid_svg.select("#imageGrid_brush_g").call(d3.brush().move, event.selection)
        imageTypeGrid_svg_right.select("#imageTypeGrid_brush_g_right").call(d3.brush().move, event.selection)
    })
let imageGrid_brush = d3.brush()
    .on("start brush", function (event) {
        heatMap_svg.select("#heatmap_brush_g").call(d3.brush().move, event.selection)
        heatMap_svg_right.select("#heatmap_brush_g_right").call(d3.brush().move, event.selection)
        imageGrid_svg_right.select("#imageGrid_brush_g_right").call(d3.brush().move, event.selection)
        imageTypeGrid_svg.select("#imageTypeGrid_brush_g").call(d3.brush().move, event.selection)
        imageTypeGrid_svg_right.select("#imageTypeGrid_brush_g_right").call(d3.brush().move, event.selection)
    })
let imageTypeGrid_brush = d3.brush()
    .on("start brush", function (event) {
        heatMap_svg.select("#heatmap_brush_g").call(d3.brush().move, event.selection)
        heatMap_svg_right.select("#heatmap_brush_g_right").call(d3.brush().move, event.selection)
        imageGrid_svg_right.select("#imageGrid_brush_g_right").call(d3.brush().move, event.selection)
        imageGrid_svg.select("#imageGrid_brush_g").call(d3.brush().move, event.selection)
        imageTypeGrid_svg_right.select("#imageTypeGrid_brush_g_right").call(d3.brush().move, event.selection)

    })
let imageTypeGrid_brush_right = d3.brush()
    .on("start brush", function (event) {
        heatMap_svg.select("#heatmap_brush_g").call(d3.brush().move, event.selection)
        heatMap_svg_right.select("#heatmap_brush_g_right").call(d3.brush().move, event.selection)
        imageGrid_svg_right.select("#imageGrid_brush_g_right").call(d3.brush().move, event.selection)
        imageGrid_svg.select("#imageGrid_brush_g").call(d3.brush().move, event.selection)
        imageTypeGrid_svg.select("#imageTypeGrid_brush_g").call(d3.brush().move, event.selection)
    })
// 监听brush button的点击事件
brushBtn.addEventListener("click", function () {
    if (brushBtn.checked) {
        // 关闭cropBtn
        if (cropBtn.checked == true) {
            cropBtn.click();
        }
        // 关闭panBtn
        if (panBtn.checked == true) {
            panBtn.click();
        }
        // 关闭locateBtn
        if (locateBtn.checked == true) {
            locateBtn.click();
        }
        // 关闭lineBtn
        if (lineBtn.checked == true) {
            lineBtn.click();
        }
        // 热力图概览添加brush
        if (document.getElementById("heatmap_brush_g") === null) {
            heatMap_svg.append("g")
                .attr("class", "brush")
                .attr("id", "heatmap_brush_g")
                .call(heatMap_brush);
        }
        if (document.getElementById("heatmap_brush_g_right") === null) {
            heatMap_svg_right.append("g")
                .attr("class", "brush")
                .attr("id", "heatmap_brush_g_right")
                .call(heatMap_brush_right);
        }
        //图片概览添加brush
        if (document.getElementById("imageGrid_brush_g") === null) {
            imageGrid_svg.append("g")
                .attr("class", "brush")
                .attr("id", "imageGrid_brush_g")
                .call(imageGrid_brush);
        }
        if (document.getElementById("imageGrid_brush_g_right") === null) {
            imageGrid_svg_right.append("g")
                .attr("class", "brush")
                .attr("id", "imageGrid_brush_g_right")
                .call(imageGrid_brush_right);
        }
        //图片类别概览添加brush
        if (document.getElementById("imageTypeGrid_brush_g") === null) {
            imageTypeGrid_svg.append("g")
                .attr("class", "brush")
                .attr("id", "imageTypeGrid_brush_g")
                .call(imageTypeGrid_brush);
        }
        if (document.getElementById("imageTypeGrid_brush_g_right") === null) {
            imageTypeGrid_svg_right.append("g")
                .attr("class", "brush")
                .attr("id", "imageTypeGrid_brush_g_right")
                .call(imageTypeGrid_brush_right);
        }
    } else {
        // 删除brush
        if (document.getElementById("heatmap_brush_g") !== null) {
            document.getElementById("heatmap_brush_g").remove();
        }
        if (document.getElementById("imageGrid_brush_g") !== null) {
            document.getElementById("imageGrid_brush_g").remove();
        }
        if (document.getElementById("imageTypeGrid_brush_g") !== null) {
            document.getElementById("imageTypeGrid_brush_g").remove();
        }
          // 删除brush
        if (document.getElementById("heatmap_brush_g_right") !== null) {
            document.getElementById("heatmap_brush_g_right").remove();
        }
        if (document.getElementById("imageGrid_brush_g_right") !== null) {
            document.getElementById("imageGrid_brush_g_right").remove();
        }
        if (document.getElementById("imageTypeGrid_brush_g_right") !== null) {
            document.getElementById("imageTypeGrid_brush_g_right").remove();
        }
    }
})

// 点击位置事件~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 显示置信度函数
// function show_confidence_tran(confidence, side) {
//
//     // console.log(index_max, index_max_2);
//     const svg_width = 180;
//     const svg_height = 280;
//     const rect_height = 17;
//     const margin = 10;
//     const shift = 13;
//     const max_width = 60
//     const min_width = 0.1
//
//     // 创建置信度直方图svg
//     const svg_confidence = d3.select("#svgContainer_probability_distribution")
//         .select("#confidence_distribution")
//         .attr("width", svg_width)
//         .attr("height", svg_height)
//     svg_confidence.select("#bars_1")
//         .attr("class", "bars1")
//         .attr("transform", "translate(4, 5)");
//     svg_confidence.select("#labels")
//         .attr("class", "labels")
//         .attr("transform", "translate(90, 5)")
//     svg_confidence.select("#bars_2")
//         .attr("class", "bars2")
//         .attr("transform", "translate(110, 5)")
//
//     if (side == "left") {
//         //添加矩形
//         d3.select('#bars_1')
//             .selectAll('rect')
//             .data(confidence)
//             .join('rect')
//             .attr('y', function (d, i) {
//                 return i * (rect_height + margin);
//             })
//             .attr('fill', 'grey')
//             .attr('height', rect_height)
//             .attr("width", 0)
//             .attr('x', function (d) {
//                 return max_width + min_width;
//             })
//             .transition()
//             .duration(2000)
//             .attr('width', function (d) {
//                 return d * max_width + min_width; //2是保底宽度
//             })
//             .attr('x', function (d) {
//                 return max_width - d * max_width;
//             })
//         //添加文字
//         d3.select('#bars_1')
//             .selectAll('text')
//             .data(confidence)
//             .join('text')
//             .attr("class", 'labels_text')
//             .attr('y', function (d, i) {
//                 return i * (rect_height + margin) + shift;
//             })
//             .text(function (d) {
//                 return d;
//             })
//             .attr('x', function (d) {
//                 return max_width + min_width;
//             })
//             .style("fill", function (d) {
//                 if (d <= 0.5) {
//                     return "black"
//                 }
//                 else {
//                     return "red"
//                 }
//             })
//             .transition()
//             .duration(2000)
//             .attr('x', function (d) {
//                 if (d <= 0.5) {
//                     return max_width - d * max_width;
//                 }
//                 else {
//                     return max_width - d * max_width + 25;
//                 }
//
//             })
//
//     } else if (side == "right") {
//         //添加矩形
//         d3.select('#bars_2')
//             .selectAll('rect')
//             .data(confidence)
//             .join('rect')
//             .attr('y', function (d, i) {
//                 return i * (rect_height + margin);
//             })
//             .attr('fill', 'grey')
//             .attr('height', rect_height)
//             .attr("width", 0)
//             .transition()
//             .duration(2000)
//             .attr('width', function (d) {
//                 return d * max_width + min_width; //2是保底宽度
//             })
//         //添加文字
//         d3.select('#bars_2')
//             .selectAll('text')
//             .data(confidence)
//             .join('text')
//             .attr("class", 'labels_text')
//             .attr("x", 0)
//             .attr('y', function (d, i) {
//                 return i * (rect_height + margin) + shift;
//             })
//             .text(function (d) {
//                 return d;
//             })
//             .style("fill", function (d) {
//                 if (d <= 0.5) {
//                     return "black"
//                 }
//                 else {
//                     return "red"
//                 }
//             })
//             .transition()
//             .duration(2000)
//             .attr('x', function (d) {
//                 if (d <= 0.5) {
//                     return d * max_width;
//                 }
//                 else {
//                     return d * max_width - 25;
//                 }
//
//             })
//     }
//
//
//     d3.select('#labels')
//         .selectAll('text')
//         .data(cifar10_classes) //cifar10_classes数组在image_type_grid.js中定义
//         .join('text')
//         .attr('y', function (d, i) {
//             return i * (rect_height + margin) + shift;
//         })
//         .text(function (d) {
//             return d;
//         })
//         .style('text-anchor', 'middle');
//
// }

function show_confidence(confidence,confidence_label,dataset_type, side) {

    // console.log(index_max, index_max_2);
    const svg_width = 400;
    const svg_height = 180;
    const rect_height = 18;
    const margin = 18;
    const shift = 13;
    const max_height = 35
    const min_height = 0.1
    if (dataset_type === "CIFAR10") {
        class_names = confidence_label.map(label => cifar10_classes[label])
    }
    if (dataset_type === "GTSRB") {
        class_names = confidence_label.map(label => GTSRB_classes[label])
    }
    // 创建置信度直方图svg
    const svg_confidence = d3.select("#svgContainer_probability_distribution")
        .select("#confidence_distribution")
        .attr("width", svg_width)
        .attr("height", svg_height)
    svg_confidence.select("#bars_1")
        .attr("class", "bars1")
        .attr("transform", "translate(20, 0)");
    svg_confidence.select("#labels_1")
        .attr("class", "labels")
        .attr("transform", "translate(20, 70)")
    svg_confidence.select("#labels_2")
        .attr("class", "labels")
        .attr("transform", "translate(20, 95)")
    svg_confidence.select("#bars_2")
        .attr("class", "bars2")
        .attr("transform", "translate(20, 125)")

    if (side == "left") {
        // d3.select('#bars_1')
        //     .selectAll('text')
        //     .remove()
        // d3.select('#bars_1')
        //     .selectAll('rect')
        //     .remove()
        // d3.select('#bars_2')
        //     .selectAll('text')
        //     .remove()
        // d3.select('#bars_2')
        //     .selectAll('rect')
        //     .remove()
        //添加矩形
        d3.select('#bars_1')
            .selectAll('rect')
            .data(confidence)
            .join('rect')
            .attr('x', function (d, i) {
                return i * (rect_height + margin);
            })
            .attr('fill', '#A1A1A1')
            .attr('width', rect_height)
            .attr("y", 70) // 初始化y为SVG底部,一定要初始化，才能有动画
            .transition()
            .duration(0)
            .attr('height', function (d) {
                return d * max_height + min_height; //2是保底宽度
            })
            .attr('y', function (d) {
                return 70 - (d * max_height + min_height);
            })
        //添加文字
        d3.select('#bars_1')
            .selectAll('text')
            .data(confidence)
            .join('text')
            .attr("class", 'labels_text')
            .attr("x", function (d, i) {
                return i * (rect_height + margin) + rect_height / 2;
            })
            .attr('y', function (d) {
                return 70 - (d * max_height + min_height) - shift; // 确保文字在条形图顶部
            })
            .text(function (d) {
                                           // 检查是否为数组以及数组是否非空
                if (Array.isArray(d) && d.length > 0) {
                    // 将数组中的第一个元素转换为数字
                    var num = parseFloat(d[0]);
                    if (!isNaN(num)) {
                        // 将数字格式化为三位小数
                        return num.toFixed(3);
                    } else {
                        // 如果无法转换为数字，返回原始值
                        return d[0];
                    }
                } else {
                    // 如果不是数组或者是空数组，返回原始值
                    return d;
                }
            })
            .style("fill", function (d) {
                if (d <= 0.5) {
                    return "black"
                }
                else {
                    return "red"
                }
            })
            .style("text-anchor", "middle")
            // .transition()
            // .duration(2000)
            .attr("transform", function(d, i) {
                // 计算旋转中心点
                var x = i * (rect_height + margin) + rect_height / 2;
                var y = 70 - (d * max_height + min_height) - shift / 2;
                return `rotate(-35, ${x}, ${y})`; // 旋转-45度
            })
            .attr("font-family", "Consolas, courier");
        if(dataset_type === "CIFAR10"){
            d3.select('#labels_1')
                .selectAll('image')
                .remove()
            d3.select('#labels_1')
                .selectAll('text')
                .data(class_names) //cifar10_classes数组在image_type_grid.js中定义
                .join('text')
                .attr('x', function (d, i) {
                    return i * (rect_height + margin) + rect_height / 2;
                })
                .text(function (d) {
                    return d;
                })
                .attr('y', 15) // 确保标签在顶部
                .style('text-anchor', 'middle');
        }
        else if(dataset_type === "GTSRB"){
            d3.select('#labels_1')
                .selectAll('text')
                .remove()
            d3.select('#labels_1')
                .selectAll('image')
                .data(confidence_label)
                .join('image')
                .attr('href', d => `../static/example/label_pic/${d}.png`)
                .attr('x', function (d, i) {
                    return i * (rect_height + margin);
                })
                .attr('y', 5) // 确保标签在顶部
                .style('width', 20) // 调整为你需要的宽度
                .style('height', 20)
                .attr("title", "logo image");
        }


    } else if (side == "right") {
        //添加矩形
        d3.select('#bars_2')
            .selectAll('rect')
            .data(confidence)
            .join('rect')
            .attr('x', function (d, i) {
                return i * (rect_height + margin);
            })
            .attr('fill', '#A1A1A1')
            .attr("width", rect_height)
            .attr('y', 0)
            // .transition()
            // .duration(0)
            .attr('height', function (d) {
                return d * max_height + min_height; //2是保底宽度
            })
             // .attr('y', function (d) {
             //    return (-5 + d * max_height + min_height); // 确保条形图从底部向上绘制
            // });
        //添加文字
        d3.select('#bars_2')
            .selectAll('text')
            .data(confidence)
            .join('text')
            .attr("class", 'labels_text')
            .attr("x", function (d, i) {
                return i * (rect_height + margin) + rect_height / 2;
            })
            .attr('y', function (d) {
                return (d * max_height + min_height) + 10;
            })
            .text(function (d) {
                                       // 检查是否为数组以及数组是否非空
                if (Array.isArray(d) && d.length > 0) {
                    // 将数组中的第一个元素转换为数字
                    var num = parseFloat(d[0]);
                    if (!isNaN(num)) {
                        // 将数字格式化为三位小数
                        return num.toFixed(3);
                    } else {
                        // 如果无法转换为数字，返回原始值
                        return d[0];
                    }
                } else {
                    // 如果不是数组或者是空数组，返回原始值
                    return d;
                }
            })
            .style("fill", function (d) {
                if (d <= 0.5) {
                    return "black"
                }
                else {
                    return "red"
                }
            })
            .style("text-anchor", "middle")
            // .transition()
            // .duration(2000)
            .attr("transform", function(d, i) {
            // 计算旋转中心点
            var x = i * (rect_height + margin) + rect_height / 2;
            var y = (d * max_height + min_height) + shift/2;
            return `rotate(-35, ${x}, ${y})`; // 旋转-45度
            })
            .attr("font-family", "Consolas, courier");
        if(dataset_type === "CIFAR10"){
            d3.select('#labels_2')
                .selectAll('image')
                .remove()
            d3.select('#labels_2')
                .selectAll('text')
                .data(class_names) //cifar10_classes数组在image_type_grid.js中定义
                .join('text')
                .attr('x', function (d, i) {
                    return i * (rect_height + margin) + rect_height / 2;
                })
                .text(function (d) {
                    return d;
                })
                .attr('y', 15) // 确保标签在顶部
                .style('text-anchor', 'middle');
        }
        else if(dataset_type === "GTSRB"){
            d3.select('#labels_2')
                .selectAll('text')
                .remove()
            d3.select('#labels_2')
                .selectAll('image')
                .data(confidence_label)
                .join('image')
                .attr('href', d => `../static/example/label_pic/${d}.png`)
                .attr('x', function (d, i) {
                    return i * (rect_height + margin);
                })
                .attr('y', 5) // 确保标签在顶部
                .style('width', 20) // 调整为你需要的宽度
                .style('height', 20)
                .attr("title", "logo image");
        }
    }


    // d3.select('#labels')
    //     .selectAll('text')
    //     .data(cifar10_classes) //cifar10_classes数组在image_type_grid.js中定义
    //     .join('text')
    //     .attr('x', function (d, i) {
    //         return i * (rect_height + margin) + rect_height / 2;
    //     })
    //     .text(function (d) {
    //         return d;
    //     })
    //     .attr('y', 15) // 确保标签在顶部
    //     .style('text-anchor', 'middle');

}
// 这里不做模型对比，就显示一个模型的信息
function show_confidence_single_tran(confidence) {

    // console.log(index_max, index_max_2);
    const svg_width = 180;
    const svg_height = 280;
    const rect_height = 17;
    const margin = 10;
    const shift = 13;
    const max_width = 60
    const min_width = 0.1

    // 创建置信度直方图svg
    const svg_confidence = d3.select("#svgContainer_probability_distribution")
        .select("#confidence_distribution")
        .attr("width", svg_width)
        .attr("height", svg_height)
    svg_confidence.select("#labels")
        .attr("class", "labels")
        .attr("transform", "translate(60, 5)")
    svg_confidence.select("#bars_2")
        .attr("class", "bars2")
        .attr("transform", "translate(80, 5)")

    //添加矩形
    d3.select('#bars_2')
        .selectAll('rect')
        .data(confidence)
        .join('rect')
        .attr('y', function (d, i) {
            return i * (rect_height + margin);
        })
        .attr('fill', 'grey')
        .attr('height', rect_height)
        .attr("width", 0)
        .transition()
        .duration(2000)
        .attr('width', function (d) {
            return d * max_width + min_width; //2是保底宽度
        })
    //添加文字
    d3.select('#bars_1')
        .selectAll('text')
        .remove()
    d3.select('#bars_2')
        .selectAll('text')
        .data(confidence)
        .join('text')
        .attr("class", 'labels_text')
        .attr("x", 0)
        .attr('y', function (d, i) {
            return i * (rect_height + margin) + shift;
        })
        .text(function (d) {
            return d;
        })
        .style("fill", function (d) {
            if (d <= 0.5) {
                return "black"
            }
            else {
                return "red"
            }
        })
        .transition()
        .duration(2000)
        .attr('x', function (d) {
            if (d <= 0.5) {
                return d * max_width;
            }
            else {
                return d * max_width - 25;
            }

        })



    d3.select('#labels')
        .selectAll('text')
        .data(cifar10_classes) //cifar10_classes数组在image_type_grid.js中定义
        .join('text')
        .attr('y', function (d, i) {
            return i * (rect_height + margin) + shift;
        })
        .text(function (d) {
            return d;
        })
        .style('text-anchor', 'middle');

}
function show_confidence_single(confidence,confidence_label,dataset_type) {

    // console.log(index_max, index_max_2);
    const svg_width = 400;
    const svg_height = 100;
    const rect_height = 18;
    const margin = 18;
    const shift = 13;
    const max_height = 35; // 调整为适应纵向高度
    const min_height = 0.1;
    if (dataset_type === "CIFAR10") {
        class_names = confidence_label.map(label => cifar10_classes[label])
    }
    if (dataset_type === "GTSRB") {
        class_names = confidence_label.map(label => GTSRB_classes[label])
    }
    // 创建置信度直方图svg
    const svg_confidence = d3.select("#svgContainer_probability_distribution")
        .select("#confidence_distribution")
        .attr("width", svg_width)
        .attr("height", svg_height)
    svg_confidence.select("#labels_1")
        .attr("class", "labels")
        .attr("transform", "translate(20, 70)")
    svg_confidence.select("#bars_1")
        .attr("class", "bars1")
        .attr("transform", "translate(20, 0)")
    d3.select('#bars_2')
        .selectAll('text')
        .remove()
    d3.select('#labels_2')
        .selectAll('image')
        .remove()
    d3.select('#labels_2')
        .selectAll('text')
        .remove()
    d3.select('#bars_2')
        .selectAll('rect')
        .remove()
    d3.select('#bars_1')
        .selectAll('text')
        .remove()
    // d3.select('#labels_1')
    //     .selectAll('image')
    //     .remove()
    // d3.select('#labels_1')
    //     .selectAll('text')
    //     .remove()
    d3.select('#bars_1')
        .selectAll('rect')
        .remove()
    //添加矩形
    d3.select('#bars_1')
        .selectAll('rect')
        .data(confidence)
        .join('rect')
        .attr('x', function (d, i) {
            return i * (rect_height + margin);
        })
        .attr('fill', '#A1A1A1')
        .attr('width', rect_height)
        .attr("y", svg_height - 30) // 初始化y为SVG底部,一定要初始化，才能有动画
        .transition()
        .duration(0)
        .attr('height', function (d) {
            return d * max_height + min_height; //2是保底宽度
        })
        .attr('y', function (d) {
            return svg_height - 30 - (d * max_height + min_height); // 确保条形图从底部向上绘制
        });
    //添加文字
    d3.select('#bars_1')
        .selectAll('text')
        .data(confidence)
        .join('text')
        .attr("class", 'labels_text')
        .attr("x", function (d, i) {
            return i * (rect_height + margin) + rect_height / 2;
        })
        .attr('y', function (d) {
            return svg_height - 30 - (d * max_height + min_height) - shift; // 确保文字在条形图顶部
        })
        .text(function (d) {
            // 检查是否为数组以及数组是否非空
            if (Array.isArray(d) && d.length > 0) {
                // 将数组中的第一个元素转换为数字
                var num = parseFloat(d[0]);
                if (!isNaN(num)) {
                    // 将数字格式化为三位小数
                    return num.toFixed(3);
                } else {
                    // 如果无法转换为数字，返回原始值
                    return d[0];
                }
            } else {
                // 如果不是数组或者是空数组，返回原始值
                return d;
            }
        })
        .style("fill", function (d) {
            if (d <= 0.5) {
                return "black"
            } else {
                return "red"
            }
        })
        .style("text-anchor", "middle")
        // .transition()
        // .duration(2000)
        .attr("transform", function (d, i) {
            // 计算旋转中心点
            var x = i * (rect_height + margin) + rect_height / 2;
            var y = svg_height - 30 - (d * max_height + min_height) - shift / 2;
            return `rotate(-45, ${x}, ${y})`; // 旋转-45度
        })
        .attr("font-family", "Consolas, courier");


    if (dataset_type === "CIFAR10") {
        d3.select('#labels_1')
            .selectAll('image')
            .remove()
        d3.select('#labels_1')
            .selectAll('text')
            .data(class_names) //cifar10_classes数组在image_type_grid.js中定义
            .join('text')
            .attr('x', function (d, i) {
                return i * (rect_height + margin) + rect_height / 2;
            })
            .text(function (d) {
                return d;
            })
            .attr('y', 15) // 确保标签在顶部
            .style('text-anchor', 'middle')
            .attr("font-family", "Consolas, courier");
    }
    else if (dataset_type === "GTSRB") {
        console.log("使用logos")
        d3.select('#labels_1')
            .selectAll('text')
            .remove()
        d3.select('#labels_1')
            .selectAll('image')
            .data(confidence_label)
            .join('image')
            .attr('href', d => `../static/example/label_pic/${d}.png`)
            .attr('x', function (d, i) {
                return i * (rect_height + margin);
            })
            .attr('y', 5) // 确保标签在顶部
            .style('width', 20) // 调整为你需要的宽度
            .style('height', 20)
            .attr("title", "logo image");

    }
}
const locateBtn = document.querySelector("#btn_locate");
locateBtn.addEventListener("click", function () {
    if (locateBtn.checked) {
        // 关闭cropBtn
        if (cropBtn.checked == true) {
            cropBtn.click();
        }
        // 关闭panBtn
        if (panBtn.checked == true) {
            panBtn.click();
        }
        // 关闭brushBtn
        if (brushBtn.checked == true) {
            brushBtn.click();
        }
        // 关闭lineBtn
        if (lineBtn.checked == true) {
            lineBtn.click();
        }
        heatMap_svg.style("cursor", "pointer");
        imageGrid_svg.style("cursor", "pointer");
        imageTypeGrid_svg.style("cursor", "pointer");
        heatMap_svg_right.style("cursor", "pointer");
        imageGrid_svg_right.style("cursor", "pointer");
        imageTypeGrid_svg_right.style("cursor", "pointer");
        // 图片框
        d3.select("#click_img_div")
            .style("display", "block")
        // 对比图片
        d3.select("#compare_img_div")
            .style("display", "none")
        // 模型说明
        d3.select("#div_model_name")
            .style("display", "none")
        // 图片对比
        d3.select("#div_compare_pic")
            .style("display", "none")

        d3.selectAll(".div_imgcard_info").style("display", "flex")
        d3.select("#seconde_type").style("display", "none")
        d3.select("#seconde_robustness").style("display", "none")

        var rectWidth = 40;
        //刚开始进来需要将之前的信息清空
        d3.select("#click_img").attr("src", "../static/example/initial_pic/None.png")
        d3.select("#cam_img").attr("src", "../static/example/initial_pic/None.png")

        // 显示类别信息
        document.getElementById("classified-as_1").innerText = ""; //cifar10_classes数组在image_type_grid.js中定义
        document.getElementById("classified-as_2").innerText = ""; //cifar10_classes数组在image_type_grid.js中定义
        document.getElementById("classified-as_1_real").innerText = ""; //cifar10_classes数组在image_type_grid.js中定义
        document.getElementById("classified-as_2_real").innerText = "";
        // 显示鲁棒性
        document.getElementById("predicted-robustness-value_1").innerText = "";
        document.getElementById("predicted-robustness-value_2").innerText = "";
        // 显示置信度
        dataset_type = document.getElementById("select_dataset_type_selection").value
        zero_confidece = [0, 0, 0, 0, 0, 0, 0, 0]
        zero_confidece_label = [0, 1, 2, 3, 4, 5, 6, 7]
        show_confidence_single(zero_confidece, zero_confidece_label,dataset_type)
        // show_confidence(zero_confidece, "left")
        // show_confidence(zero_confidece, "right")
        //热力图绑定点击事件
        heatMap_svg.on("click", async function (event) {
            var temp_pos = d3.pointer(event) //获取事件坐标
            // console.log("temp_pos: ", temp_pos)
            // var x = xScale.invert(temp_pos[0])
            // var y = yScale.invert(temp_pos[1])
            var x = offset_xScale_invert(xScale, temp_pos[0])
            var y = offset_yScale_invert(yScale, temp_pos[1])

            //在热力图上添加点
            heatMap_svg.select("#imageAndCircle_g")
                .append("circle")
                .attr("cx", temp_pos[0])
                .attr("cy", temp_pos[1])
                .attr("pos_x", x)
                .attr("pos_y", y)
                .attr("r", "3")
                .attr("fill", "black")
                .attr("fill-opacity", "1")
            img_number = img_number + 1;
            heatMap_svg.style("cursor", "wait");
            imageGrid_svg.style("cursor", "wait");
            imageTypeGrid_svg.style("cursor", "wait");

            console.log("点击热力图时候的坐标： " +String(x) + ", " + String(y))
            var img_information = await get_image_information_from_python(x = x, y = y, img_name = String(img_number),img_type = 0)
            // 在热力图上添加图片
            console.log("点击热力图时候返回的信息： ", img_information)
            heatMap_svg.select("#imageAndCircle_g")
                .append("image")
                .attr("href", "../static/example/pic/" + String(img_number) + ".png?t=" + Math.random())
                .attr("x", temp_pos[0])
                .attr("y", temp_pos[1])
                .attr("pos_x", x)
                .attr("pos_y", y)
                .attr("width", rectWidth)
                .attr("height", rectWidth)

            // 修改左边示例图
            d3.select("#click_img").attr("src", "../static/example/pic/" + String(img_number) + ".png?t=" + Math.random())
            // 左边部分
            var dataset_type = document.getElementById("select_dataset_type_selection").value
            if (dataset_type == "CIFAR10") {
                // 显示类别信息
                document.getElementById("classified-as_1").innerText = cifar10_classes[parseInt(img_information['label']['M1'])]; //cifar10_classes数组在
            } else if (dataset_type == "SteeringAngle") {
                // 显示类别信息
                document.getElementById("classified-as_1").innerText = ((img_information['label']['M1'] - 0.5) * 160).toFixed(); //cifar10_classes数组在image_type_grid.js中定义
            } 
            // 显示鲁棒性
            document.getElementById("predicted-robustness-value_1").innerText = (img_information['img_robustness']["M1"] * 1000).toFixed(0);
            // 显示置信度
            // show_confidence(img_information['layer']["M1"], "left")
            show_confidence_single(img_information['layer']["M1"])
            // 右边部分
            var model_name = document.getElementById("select_model_selection_compare").value
            if (model_name != "None") {
                // 显示类别信息
                document.getElementById("classified-as_2").innerText = cifar10_classes[parseInt(img_information['label']["M2"])]; //cifar10_classes数组在image_type_grid.js中定义
                // 显示鲁棒性
                document.getElementById("predicted-robustness-value_2").innerText = (img_information['img_robustness']["M2"] * 1000).toFixed(0);
                // 显示置信度
                show_confidence(img_information['layer']["M2"], "right")
            }


            heatMap_svg.style("cursor", "pointer");
            imageGrid_svg.style("cursor", "pointer");
            imageTypeGrid_svg.style("cursor", "pointer");
        })
        heatMap_svg_right.on("click", async function (event) {
            var temp_pos = d3.pointer(event) //获取事件坐标
            // console.log("temp_pos: ", temp_pos)
            // var x = xScale.invert(temp_pos[0])
            // var y = yScale.invert(temp_pos[1])
            var x = offset_xScale_invert(xScale, temp_pos[0])
            var y = offset_yScale_invert(yScale, temp_pos[1])

            //在热力图上添加点
            heatMap_svg_right.select("#imageAndCircle_g_right")
                .append("circle")
                .attr("cx", temp_pos[0])
                .attr("cy", temp_pos[1])
                .attr("pos_x", x)
                .attr("pos_y", y)
                .attr("r", "3")
                .attr("fill", "black")
                .attr("fill-opacity", "1")
            img_number = img_number + 1;
            heatMap_svg_right.style("cursor", "wait");
            imageGrid_svg_right.style("cursor", "wait");
            imageTypeGrid_svg_right.style("cursor", "wait");

            console.log("点击热力图时候的坐标： " +String(x) + ", " + String(y))
            var img_information = await get_image_information_from_python(x = x, y = y, img_name = String(img_number),img_type = 1)
            // 在热力图上添加图片
            console.log("点击热力图时候返回的信息： ", img_information)
            heatMap_svg_right.select("#imageAndCircle_g_right")
                .append("image")
                .attr("href", "../static/example/pic/" + String(img_number) + ".png?t=" + Math.random())
                .attr("x", temp_pos[0])
                .attr("y", temp_pos[1])
                .attr("pos_x", x)
                .attr("pos_y", y)
                .attr("width", rectWidth)
                .attr("height", rectWidth)

            // 修改左边示例图
            d3.select("#click_img").attr("src", "../static/example/pic/" + String(img_number) + ".png?t=" + Math.random())
            d3.select("#cam_img").attr("src", "../static/example/pic/" + String(img_number) + "_cam.png?t=" + Math.random())
            // 左边部分
            // var dataset_type = document.getElementById("select_dataset_type_selection").value
            if (dataset_type === "CIFAR10") {
                // 显示类别信息
                // document.getElementById("classified-as_1").innerText = cifar10_classes[parseInt(img_information['label']['M1'])]; //cifar10_classes数组在
                document.getElementById("classified-as_1").innerText = `Pre : ${cifar10_classes[parseInt(img_information['label']['M1'])]}`;//cifar10_classes数组在image_type_grid.js中定义
                document.getElementById("classified-as_1_real").innerText = `Real: ${cifar10_classes[parseInt(img_information['real_label']['M1'])]}`;//cifar10_classes数组在image_type_grid.js中定义
                document.getElementById("classified-as_1").setAttribute('title', `Pre : ${cifar10_classes[parseInt(img_information['label']['M1'])]}`);//cifar10_classes数组在image_type_grid.js中定义
                document.getElementById("classified-as_1_real").setAttribute('title', `Real: ${cifar10_classes[parseInt(img_information['real_label']['M1'])]}`);//cifar10_classes数组在image_type_grid.js中定义

            } else if (dataset_type === "SteeringAngle") {
                // 显示类别信息
                document.getElementById("classified-as_1").innerText = ((img_information['label']['M1'] - 0.5) * 160).toFixed(); //cifar10_classes数组在image_type_grid.js中定义
            }else if (dataset_type === "GTSRB") {
                // 显示类别信息
                // document.getElementById("classified-as_1").innerText = GTSRB_classes[parseInt(img_information['label']['M1'])];
                document.getElementById("classified-as_1").innerText = `Pre : ${GTSRB_classes[parseInt(img_information['label']['M1'])]}`;//cifar10_classes数组在image_type_grid.js中定义
                document.getElementById("classified-as_1_real").innerText = `Real: ${GTSRB_classes[parseInt(img_information['real_label']['M1'])]}`;//cifar10_classes数组在image_type_grid.js中定义
                document.getElementById("classified-as_1").setAttribute('title', `Pre : ${GTSRB_classes[parseInt(img_information['label']['M1'])]}`);//cifar10_classes数组在image_type_grid.js中定义
                document.getElementById("classified-as_1_real").setAttribute('title', `Real: ${GTSRB_classes[parseInt(img_information['real_label']['M1'])]}`);//cifar10_classes数组在image_type_grid.js中定义

            }

            // 显示鲁棒性
            document.getElementById("predicted-robustness-value_1").innerText = (img_information['img_robustness']["M1"] * 1000).toFixed(0);
            // 显示置信度
            // show_confidence(img_information['layer']["M1"], "left")
            show_confidence_single(img_information['layer']["M1"],img_information['layer_label']["M1"],dataset_type)
            // 右边部分
            var model_name = document.getElementById("select_model_selection_compare").value
            if (model_name != "None") {
                // 显示类别信息
                document.getElementById("classified-as_2").innerText = cifar10_classes[parseInt(img_information['label']["M2"])]; //cifar10_classes数组在image_type_grid.js中定义
                // 显示鲁棒性
                document.getElementById("predicted-robustness-value_2").innerText = (img_information['img_robustness']["M2"] * 1000).toFixed(0);
                // 显示置信度
                show_confidence(img_information['layer']["M2"], "right")
            }


            heatMap_svg_right.style("cursor", "pointer");
            imageGrid_svg_right.style("cursor", "pointer");
            imageTypeGrid_svg_right.style("cursor", "pointer");
        })
        //图片概览添加点击事件
        var imageGrid_g = imageGrid_svg.select("#imageGrid_g");
        imageGrid_g.selectAll("image")
            .on("click", async function (event) {
                //当前点击图片的序号
                var img_number = d3.select(this).attr("number");
                //当前点击图片的坐标(原始的坐标)
                var img_coordinate = d3.select(this).data()[0];
                console.log("img_number: ", img_number)
                console.log("img_coordinate: ", img_coordinate)
                heatMap_svg.style("cursor", "wait");
                imageGrid_svg.style("cursor", "wait");
                imageTypeGrid_svg.style("cursor", "wait");
                var img_information = await evaluate_image_from_python(img_number = img_number, img_name = String(img_number),img_type =0)
                // console.log("单机图片后返回的信息：", img_information)
                // 在热力图上添加图片
                heatMap_svg.select("#imageAndCircle_g")
                    .append("image")
                    .attr("href", "../static/example/pic/" + String(img_number) + ".png?t=" + Math.random())
                    .attr("x", offset_xScale(xScale, img_coordinate[0])) //获得偏移后的前端坐标，再main.html中定义
                    .attr("y", offset_yScale(yScale, img_coordinate[1]))
                    .attr("pos_x", img_coordinate[0])
                    .attr("pos_y", img_coordinate[1])
                    .attr("width", rectWidth)
                    .attr("height", rectWidth)
                //在热力图上添加点
                heatMap_svg.select("#imageAndCircle_g")
                    .append("circle")
                    .attr("cx", offset_xScale(xScale, img_coordinate[0])) //获得偏移后的前端坐标，再main.html中定义
                    .attr("cy", offset_yScale(yScale, img_coordinate[1]))
                    .attr("pos_x", img_coordinate[0])
                    .attr("pos_y", img_coordinate[1])
                    .attr("r", "3")
                    .attr("fill", "black")
                    .attr("fill-opacity", "1")
                // 修改左边示例图
                d3.select("#click_img").attr("src", "../static/example/pic/" + String(img_number) + ".png?t=" + Math.random())
                var dataset_type = document.getElementById("select_dataset_type_selection").value
                if (dataset_type == "CIFAR10") {
                    // 显示类别信息
                    document.getElementById("classified-as_1").innerText = cifar10_classes[parseInt(img_information['label']['M1'])]; //cifar10_classes数组在
                } else if (dataset_type == "SteeringAngle") {
                    // 显示类别信息
                    document.getElementById("classified-as_1").innerText = ((img_information['label']['M1'] - 0.5) * 160).toFixed(); //cifar10_classes数组在image_type_grid.js中定义
                }
                // 显示鲁棒性
                document.getElementById("predicted-robustness-value_1").innerText = (parseFloat(img_information['img_robustness']['M1']) * 1000).toFixed(0);
                // 显示置信度
                // show_confidence(img_information['layer']['M1'], "left")
                show_confidence_single(img_information['layer']["M1"])
                // 右边部分
                var model_name = document.getElementById("select_model_selection_compare").value
                if (model_name != "None") {
                    // 显示类别信息
                    document.getElementById("classified-as_2").innerText = cifar10_classes[parseInt(img_information['label']["M2"])]; //cifar10_classes数组在image_type_grid.js中定义
                    // 显示鲁棒性
                    document.getElementById("predicted-robustness-value_2").innerText = (parseFloat(img_information['img_robustness']["M2"]) * 1000).toFixed(0);
                    // 显示置信度
                    show_confidence(img_information['layer']["M2"], "right")
                }
                heatMap_svg.style("cursor", "pointer");
                imageGrid_svg.style("cursor", "pointer");
                imageTypeGrid_svg.style("cursor", "pointer");
            })
        var imageGrid_g_right = imageGrid_svg_right.select("#imageGrid_g_right");
        imageGrid_g_right.selectAll("image")
            .on("click", async function (event) {
                //当前点击图片的序号
                var img_number = d3.select(this).attr("number");
                //当前点击图片的坐标(原始的坐标)
                var img_coordinate = d3.select(this).data()[0];
                console.log("img_coordinate: ", img_coordinate)
                heatMap_svg_right.style("cursor", "wait");
                imageGrid_svg_right.style("cursor", "wait");
                imageTypeGrid_svg_right.style("cursor", "wait");
                var img_information = await evaluate_image_from_python(img_number = img_number, img_name = String(img_number),img_type = 1)
                // console.log("单机图片后返回的信息：", img_information)
                // 在热力图上添加图片
                heatMap_svg_right.select("#imageAndCircle_g_right")
                    .append("image")
                    .attr("href", "../static/example/pic/" + String(img_number) + ".png?t=" + Math.random())
                    .attr("x", offset_xScale(xScale, img_coordinate[0])) //获得偏移后的前端坐标，再main.html中定义
                    .attr("y", offset_yScale(yScale, img_coordinate[1]))
                    .attr("pos_x", img_coordinate[0])
                    .attr("pos_y", img_coordinate[1])
                    .attr("width", rectWidth)
                    .attr("height", rectWidth)
                //在热力图上添加点
                heatMap_svg_right.select("#imageAndCircle_g_right")
                    .append("circle")
                    .attr("cx", offset_xScale(xScale, img_coordinate[0])) //获得偏移后的前端坐标，再main.html中定义
                    .attr("cy", offset_yScale(yScale, img_coordinate[1]))
                    .attr("pos_x", img_coordinate[0])
                    .attr("pos_y", img_coordinate[1])
                    .attr("r", "3")
                    .attr("fill", "black")
                    .attr("fill-opacity", "1")
                // 修改左边示例图
                d3.select("#click_img").attr("src", "../static/example/pic/" + String(img_number) + ".png?t=" + Math.random())
                d3.select("#cam_img").attr("src", "../static/example/pic/" + String(img_number) + "_cam.png?t=" + Math.random())

                var dataset_type = document.getElementById("select_dataset_type_selection").value
                if (dataset_type === "CIFAR10") {
                // 显示类别信息
                // document.getElementById("classified-as_1").innerText = cifar10_classes[parseInt(img_information['label']['M1'])]; //cifar10_classes数组在
                document.getElementById("classified-as_1").innerText = `Pre : ${cifar10_classes[parseInt(img_information['label']['M1'])]}`;//cifar10_classes数组在image_type_grid.js中定义
                document.getElementById("classified-as_1_real").innerText = `Real: ${cifar10_classes[parseInt(img_information['real_label']['M1'])]}`;//cifar10_classes数组在image_type_grid.js中定义
                document.getElementById("classified-as_1").setAttribute('title', `Pre : ${cifar10_classes[parseInt(img_information['label']['M1'])]}`);//cifar10_classes数组在image_type_grid.js中定义
                document.getElementById("classified-as_1_real").setAttribute('title', `Real: ${cifar10_classes[parseInt(img_information['real_label']['M1'])]}`);//cifar10_classes数组在image_type_grid.js中定义

                } else if (dataset_type === "SteeringAngle") {
                    // 显示类别信息
                    document.getElementById("classified-as_1").innerText = ((img_information['label']['M1'] - 0.5) * 160).toFixed(); //cifar10_classes数组在image_type_grid.js中定义
                }else if (dataset_type === "GTSRB") {
                    // 显示类别信息
                    // document.getElementById("classified-as_1").innerText = GTSRB_classes[parseInt(img_information['label']['M1'])];
                    document.getElementById("classified-as_1").innerText = `Pre : ${GTSRB_classes[parseInt(img_information['label']['M1'])]}`;//cifar10_classes数组在image_type_grid.js中定义
                    document.getElementById("classified-as_1_real").innerText = `Real: ${GTSRB_classes[parseInt(img_information['real_label']['M1'])]}`;//cifar10_classes数组在image_type_grid.js中定义
                    document.getElementById("classified-as_1").setAttribute('title', `Pre : ${GTSRB_classes[parseInt(img_information['label']['M1'])]}`);//cifar10_classes数组在image_type_grid.js中定义
                    document.getElementById("classified-as_1_real").setAttribute('title', `Real: ${GTSRB_classes[parseInt(img_information['real_label']['M1'])]}`);//cifar10_classes数组在image_type_grid.js中定义

                }
                // 显示鲁棒性
                document.getElementById("predicted-robustness-value_1").innerText = (parseFloat(img_information['img_robustness']['M1']) * 1000).toFixed(0);
                // 显示置信度
                // show_confidence(img_information['layer']['M1'], "left")
                show_confidence_single(img_information['layer']["M1"],img_information['layer_label']["M1"], dataset_type)
                // 右边部分
                var model_name = document.getElementById("select_model_selection_compare").value
                if (model_name != "None") {
                    // 显示类别信息
                    document.getElementById("classified-as_2").innerText = cifar10_classes[parseInt(img_information['label']["M2"])]; //cifar10_classes数组在image_type_grid.js中定义
                    // 显示鲁棒性
                    document.getElementById("predicted-robustness-value_2").innerText = (parseFloat(img_information['img_robustness']["M2"]) * 1000).toFixed(0);
                    // 显示置信度
                    show_confidence(img_information['layer']["M2"], "right")
                }
                heatMap_svg_right.style("cursor", "pointer");
                imageGrid_svg_right.style("cursor", "pointer");
                imageTypeGrid_svg_right.style("cursor", "pointer");
            })
        // 类别概览添加点击事件
        var imageTypeGrid_g = imageTypeGrid_svg.select("#imageTypeGrid_g")
        imageTypeGrid_g.selectAll("rect")
            .on("click", async function (event) {
                //当前点击图片的序号
                var img_number = d3.select(this).attr("number");
                //当前点击图片的坐标(原始的坐标)
                var img_coordinate = img_coords_lst_400_dict["M1"][Number(img_number)];
                console.log("img_coordinate: ", img_coordinate)
                heatMap_svg.style("cursor", "wait");
                imageGrid_svg.style("cursor", "wait");
                imageTypeGrid_svg.style("cursor", "wait");
                var img_information = await evaluate_image_from_python(img_number = img_number, img_name = String(img_number),img_type =0)
                console.log("单击类别后返回的信息：", img_information)
                // 在热力图上添加图片
                heatMap_svg.select("#imageAndCircle_g")
                    .append("image")
                    .attr("href", "../static/example/pic/" + String(img_number) + ".png?t=" + Math.random())
                    .attr("x", offset_xScale(xScale, img_coordinate[0])) // 获得偏移后的二维潜空间坐标，再main.html中定义
                    .attr("y", offset_yScale(yScale, img_coordinate[1]))
                    .attr("pos_x", img_coordinate[0]) // 通过这个确定他们在热力图种的唯一位置
                    .attr("pos_y", img_coordinate[1])
                    .attr("width", rectWidth)
                    .attr("height", rectWidth)
                //在热力图上添加点
                heatMap_svg.select("#imageAndCircle_g")
                    .append("circle")
                    .attr("cx", offset_xScale(xScale, img_coordinate[0])) // 获得偏移后的二维潜空间坐标，再main.html中定义
                    .attr("cy", offset_yScale(yScale, img_coordinate[1]))
                    .attr("pos_x", img_coordinate[0])
                    .attr("pos_y", img_coordinate[1])
                    .attr("r", "3")
                    .attr("fill", "#black")
                    .attr("fill-opacity", "1")
                // 修改左边示例图
                d3.select("#click_img").attr("src", "../static/example/pic/" + String(img_number) + ".png?t=" + Math.random())
                var dataset_type = document.getElementById("select_dataset_type_selection").value
                if (dataset_type == "CIFAR10") {
                    // 显示类别信息
                    document.getElementById("classified-as_1").innerText = cifar10_classes[parseInt(img_information['label']['M1'])]; //cifar10_classes数组在
                } else if (dataset_type == "SteeringAngle") {
                    // 显示类别信息
                    document.getElementById("classified-as_1").innerText = ((img_information['label']['M1'] - 0.5) * 160).toFixed(); //cifar10_classes数组在image_type_grid.js中定义
                }
                // 显示鲁棒性
                document.getElementById("predicted-robustness-value_1").innerText = (img_information['img_robustness']["M1"] * 1000).toFixed(0);
                // 显示置信度
                // show_confidence(img_information['layer']["M1"], "left")
                show_confidence_single(img_information['layer']["M1"])
                // 右边部分
                var model_name = document.getElementById("select_model_selection_compare").value
                if (model_name != "None") {
                    // 显示类别信息
                    document.getElementById("classified-as_2").innerText = cifar10_classes[parseInt(img_information['label']["M2"])]; //cifar10_classes数组在image_type_grid.js中定义
                    // 显示鲁棒性
                    document.getElementById("predicted-robustness-value_2").innerText = (parseFloat(img_information['img_robustness']["M2"]) * 1000).toFixed(0);
                    // 显示置信度
                    show_confidence(img_information['layer']["M2"], "right")
                }

                heatMap_svg.style("cursor", "pointer");
                imageGrid_svg.style("cursor", "pointer");
                imageTypeGrid_svg.style("cursor", "pointer");
            })
        var imageTypeGrid_g_right = imageTypeGrid_svg_right.select("#imageTypeGrid_g_right")
        imageTypeGrid_g_right.selectAll("rect")
            .on("click", async function (event) {
                //当前点击图片的序号
                var img_number = d3.select(this).attr("number");
                //当前点击图片的坐标(原始的坐标)
                var img_coordinate = img_coords_lst_400_dict["M1"][Number(img_number)];
                console.log("img_coordinate: ", img_coordinate)
                heatMap_svg_right.style("cursor", "wait");
                imageGrid_svg_right.style("cursor", "wait");
                imageTypeGrid_svg_right.style("cursor", "wait");
                var img_information = await evaluate_image_from_python(img_number = img_number, img_name = String(img_number),img_type =1)
                console.log("单击类别后返回的信息：", img_information)
                // 在热力图上添加图片
                heatMap_svg_right.select("#imageAndCircle_g_right")
                    .append("image")
                    .attr("href", "../static/example/pic/" + String(img_number) + ".png?t=" + Math.random())
                    .attr("x", offset_xScale(xScale, img_coordinate[0])) // 获得偏移后的二维潜空间坐标，再main.html中定义
                    .attr("y", offset_yScale(yScale, img_coordinate[1]))
                    .attr("pos_x", img_coordinate[0]) // 通过这个确定他们在热力图种的唯一位置
                    .attr("pos_y", img_coordinate[1])
                    .attr("width", rectWidth)
                    .attr("height", rectWidth)
                //在热力图上添加点
                heatMap_svg_right.select("#imageAndCircle_g_right")
                    .append("circle")
                    .attr("cx", offset_xScale(xScale, img_coordinate[0])) // 获得偏移后的二维潜空间坐标，再main.html中定义
                    .attr("cy", offset_yScale(yScale, img_coordinate[1]))
                    .attr("pos_x", img_coordinate[0])
                    .attr("pos_y", img_coordinate[1])
                    .attr("r", "3")
                    .attr("fill", "#black")
                    .attr("fill-opacity", "1")
                // 修改左边示例图
                d3.select("#click_img").attr("src", "../static/example/pic/" + String(img_number) + ".png?t=" + Math.random())
                d3.select("#cam_img").attr("src", "../static/example/pic/" + String(img_number) + "_cam.png?t=" + Math.random())

                var dataset_type = document.getElementById("select_dataset_type_selection").value
                if (dataset_type === "CIFAR10") {
                    // 显示类别信息
                    // document.getElementById("classified-as_1").innerText = cifar10_classes[parseInt(img_information['label']['M1'])]; //cifar10_classes数组在
                    document.getElementById("classified-as_1").innerText = `Pre : ${cifar10_classes[parseInt(img_information['label']['M1'])]}`;//cifar10_classes数组在image_type_grid.js中定义
                    document.getElementById("classified-as_1_real").innerText = `Real: ${cifar10_classes[parseInt(img_information['real_label']['M1'])]}`;//cifar10_classes数组在image_type_grid.js中定义
                    document.getElementById("classified-as_1").setAttribute('title', `Pre : ${cifar10_classes[parseInt(img_information['label']['M1'])]}`);//cifar10_classes数组在image_type_grid.js中定义
                    document.getElementById("classified-as_1_real").setAttribute('title', `Real: ${cifar10_classes[parseInt(img_information['real_label']['M1'])]}`);//cifar10_classes数组在image_type_grid.js中定义

                } else if (dataset_type === "SteeringAngle") {
                    // 显示类别信息
                    document.getElementById("classified-as_1").innerText = ((img_information['label']['M1'] - 0.5) * 160).toFixed(); //cifar10_classes数组在image_type_grid.js中定义
                }else if (dataset_type === "GTSRB") {
                    // 显示类别信息
                    // document.getElementById("classified-as_1").innerText = GTSRB_classes[parseInt(img_information['label']['M1'])];
                    document.getElementById("classified-as_1").innerText = `Pre : ${GTSRB_classes[parseInt(img_information['label']['M1'])]}`;//cifar10_classes数组在image_type_grid.js中定义
                    document.getElementById("classified-as_1_real").innerText = `Real: ${GTSRB_classes[parseInt(img_information['real_label']['M1'])]}`;//cifar10_classes数组在image_type_grid.js中定义
                    document.getElementById("classified-as_1").setAttribute('title', `Pre : ${GTSRB_classes[parseInt(img_information['label']['M1'])]}`);//cifar10_classes数组在image_type_grid.js中定义
                    document.getElementById("classified-as_1_real").setAttribute('title', `Real: ${GTSRB_classes[parseInt(img_information['real_label']['M1'])]}`);//cifar10_classes数组在image_type_grid.js中定义

                }
                // 显示鲁棒性
                document.getElementById("predicted-robustness-value_1").innerText = (img_information['img_robustness']["M1"] * 1000).toFixed(0);
                // 显示置信度
                // show_confidence(img_information['layer']["M1"], "left")
                show_confidence_single(img_information['layer']["M1"],img_information['layer_label']["M1"],dataset_type)
                // 右边部分
                var model_name = document.getElementById("select_model_selection_compare").value
                if (model_name != "None") {
                    // 显示类别信息
                    document.getElementById("classified-as_2").innerText = cifar10_classes[parseInt(img_information['label']["M2"])]; //cifar10_classes数组在image_type_grid.js中定义
                    // 显示鲁棒性
                    document.getElementById("predicted-robustness-value_2").innerText = (parseFloat(img_information['img_robustness']["M2"]) * 1000).toFixed(0);
                    // 显示置信度
                    show_confidence(img_information['layer']["M2"], "right")
                }

                heatMap_svg_right.style("cursor", "pointer");
                imageGrid_svg_right.style("cursor", "pointer");
                imageTypeGrid_svg_right.style("cursor", "pointer");
            })
    } else {
        heatMap_svg.style("cursor", "default");
        imageGrid_svg.style("cursor", "default");
        imageTypeGrid_svg.style("cursor", "default");
        //取消点击事件
        heatMap_svg.on("click", null)
        imageGrid_svg.on("click", null)
        imageTypeGrid_svg.on("click", null)

        heatMap_svg_right.style("cursor", "default");
        imageGrid_svg_right.style("cursor", "default");
        imageTypeGrid_svg_right.style("cursor", "default");
        //取消点击事件
        heatMap_svg_right.on("click", null)
        imageGrid_svg_right.on("click", null)
        imageTypeGrid_svg_right.on("click", null)

    }
});

// 画线对比事件~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
const lineBtn = document.querySelector("#btn_line");
var icon_size = 16; //标记图标的大小
lineBtn.addEventListener("click", function () {
    if (lineBtn.checked) {
        // 关闭cropBtn
        if (cropBtn.checked == true) {
            cropBtn.click();
        }
        // 关闭panBtn
        if (panBtn.checked == true) {
            panBtn.click();
        }
        // 关闭brushBtn
        if (brushBtn.checked == true) {
            brushBtn.click();
        }
        // 关闭locateBtn
        if (locateBtn.checked == true) {
            locateBtn.click();
        }
        heatMap_svg_right.style("cursor", "pointer");
        imageGrid_svg_right.style("cursor", "pointer");
        imageTypeGrid_svg_right.style("cursor", "pointer");
        d3.select("#click_img_div")
            .style("display", "none")
        d3.select("#compare_img_div")
            .style("display", "block")
        d3.select("#div_model_name")
            .style("display", "none")
        d3.select("#div_compare_pic")
            .style("display", "flex")
        //刚开始进来需要将之前的信息清空
        d3.select("#click_start").attr("src", "../static/example/initial_pic/None.png")
        d3.select("#cam_start").attr("src", "../static/example/initial_pic/None.png")

        d3.select("#click_end").attr("src", "../static/example/initial_pic/None.png")
        d3.select("#cam_end").attr("src", "../static/example/initial_pic/None.png")


        // 显示类别信息
        document.getElementById("classified-as_1").innerText = ""; //cifar10_classes数组在image_type_grid.js中定义
        document.getElementById("classified-as_2").innerText = ""; //cifar10_classes数组在image_type_grid.js中定义
        document.getElementById("classified-as_1_real").innerText = ""; //cifar10_classes数组在image_type_grid.js中定义
        document.getElementById("classified-as_2_real").innerText = "";
        // 显示鲁棒性
        document.getElementById("predicted-robustness-value_1").innerText = "";
        document.getElementById("predicted-robustness-value_2").innerText = "";
        // 显示置信度
        dataset_type = document.getElementById("select_dataset_type_selection").value

        zero_confidece = [0, 0, 0, 0, 0, 0, 0, 0]
        zero_confidece_label = [0, 1, 2, 3, 4, 5, 6, 7]
        show_confidence(zero_confidece, zero_confidece_label,dataset_type,"left")
        show_confidence(zero_confidece, zero_confidece_label,dataset_type,"right")

        //热力图事件绑定
        var click_number = 0; //记录点击次数
        var firts_click_pos; //第一次点击的坐标
        heatMap_svg_right.on("click", async function (event) {
            var temp_pos = d3.pointer(event) //获取事件坐标
            // var x = xScale.invert(temp_pos[0])
            // var y = yScale.invert(temp_pos[1])
            var x = offset_xScale_invert(xScale, temp_pos[0]) // 获得偏移后的潜空间2维坐标，再main.html中定义
            var y = offset_yScale_invert(yScale, temp_pos[1])
            d3.selectAll(".div_imgcard_info").style("display", "flex")
            d3.select("#seconde_type").style("display", "flex")
            d3.select("#seconde_robustness").style("display", "inline-block")
            //这个if else主要用来控制线
            if (click_number == 0) {
                //刚开始进来需要将之前的信息清空
                d3.select("#click_start").attr("src", "../static/example/initial_pic/None.png")
                d3.select("#cam_start").attr("src", "../static/example/initial_pic/None.png")

                d3.select("#click_end").attr("src", "../static/example/initial_pic/None.png")
                d3.select("#cam_end").attr("src", "../static/example/initial_pic/None.png")

                // 显示类别信息
                document.getElementById("classified-as_1").innerText = ""; //cifar10_classes数组在image_type_grid.js中定义
                document.getElementById("classified-as_2").innerText = ""; //cifar10_classes数组在image_type_grid.js中定义
                document.getElementById("classified-as_1_real").innerText = ""; //cifar10_classes数组在image_type_grid.js中定义
                document.getElementById("classified-as_2_real").innerText = "";
                // 显示鲁棒性
                document.getElementById("predicted-robustness-value_1").innerText = "";
                document.getElementById("predicted-robustness-value_2").innerText = "";
                // 显示置信度
                zero_confidece = [0, 0, 0, 0, 0, 0, 0, 0]
                zero_confidece_label = [0, 1, 2, 3, 4, 5, 6, 7]
                show_confidence(zero_confidece,zero_confidece_label,dataset_type, "left")
                show_confidence(zero_confidece, zero_confidece_label, dataset_type,"right")

                click_number = click_number + 1;
                firts_click_pos = d3.pointer(event)
                // 第一次点击就添加线，刚开始起点和终点一样
                heatMap_svg_right.select("#imageCompare_g_right")
                    .append("line")
                    .attr("x1", firts_click_pos[0])
                    .attr("y1", firts_click_pos[1])
                    .attr("x2", firts_click_pos[0])
                    .attr("y2", firts_click_pos[1])
                    .attr("pos_x1", x)
                    .attr("pos_y1", y)
                    .attr("class", "kjl_line") //在my_style.css中定义
                    .attr("id", "pos" + String(firts_click_pos[0]).replace(".", "") + "" + String(firts_click_pos[1]).replace(".", ""))//把点击的坐标当作线段的id
                var click_pos = d3.pointer(event) //获取事件坐标
                // 添加点
                heatMap_svg_right.select("#imageCompare_g_right")
                    .append("circle")
                    .attr("cx", click_pos[0])
                    .attr("cy", click_pos[1])
                    .attr("pos_x", x)
                    .attr("pos_y", y)
                    .attr("r", "3")
                    .attr("fill", "#00000")
                    .attr("fill-opacity", "1")
                // 添加图例
                heatMap_svg_right.select("#imageCompare_g_right")
                    .append("g")
                    // .attr("class", "bi bi-geo-alt")
                    .attr("width", icon_size)
                    .attr("height", icon_size)
                    .append("path")
                    .attr("d", "M8 16s6-5.686 6-10A6 6 0 0 0 2 6c0 4.314 6 10 6 10zm0-7a3 3 0 1 1 0-6 3 3 0 0 1 0 6z")
                    .attr("pos_x", x)
                    .attr("pos_y", y)
                    .attr("transform", "translate(" + String(Number(click_pos[0]) - icon_size / 2) + "," + String(Number(click_pos[1]) - icon_size - 8) + ")")
                    .transition()
                    .duration(800)
                    .attr("transform", "translate(" + String(Number(click_pos[0]) - icon_size / 2) + "," + String(Number(click_pos[1]) - icon_size) + ")")
                // 如果是第一次点击, 则绑定悬停事件
                heatMap_svg_right.on("mousemove", async function (event) {
                    var move_pos = d3.pointer(event)
                    heatMap_svg_right.select("#imageCompare_g_right")
                        .select("#" + "pos" + String(firts_click_pos[0]).replace(".", "") + "" + String(firts_click_pos[1]).replace(".", ""))
                        .attr("x2", move_pos[0])
                        .attr("y2", move_pos[1])
                        .attr("pos_x2", offset_xScale_invert(xScale, move_pos[0])) // 保存偏移后的终点二维潜空间坐标，再main.js中实现
                        .attr("pos_y2", offset_yScale_invert(yScale, move_pos[1]))
                })
            } else {
                click_number = 0;
                //如果不是第一次点击了，就取消悬停事件
                heatMap_svg_right.on("mousemove", null)
                var click_pos = d3.pointer(event) //获取事件坐标
                // 添加点
                heatMap_svg_right.select("#imageCompare_g_right")
                    .append("circle")
                    .attr("cx", click_pos[0])
                    .attr("cy", click_pos[1])
                    .attr("pos_x", x)
                    .attr("pos_y", y)
                    .attr("r", "3")
                    .attr("fill", "#00000")
                    .attr("fill-opacity", "1")
                // 添加图例
                d = ["M12.166 8.94c-.524 1.062-1.234 2.12-1.96 3.07A31.493 31.493 0 0 1 8 14.58a31.481 31.481 0 0 1-2.206-2.57c-.726-.95-1.436-2.008-1.96-3.07C3.304 7.867 3 6.862 3 6a5 5 0 0 1 10 0c0 .862-.305 1.867-.834 2.94zM8 16s6-5.686 6-10A6 6 0 0 0 2 6c0 4.314 6 10 6 10z", "M8 8a2 2 0 1 1 0-4 2 2 0 0 1 0 4zm0 1a3 3 0 1 0 0-6 3 3 0 0 0 0 6z"]
                heatMap_svg_right.select("#imageCompare_g_right")
                    .append("g")
                    .attr("width", icon_size)
                    .attr("height", icon_size)
                    .selectAll("path")
                    .data(d)
                    .join("path")
                    .attr("d", d => d)
                    .attr("pos_x", x)
                    .attr("pos_y", y)
                    .attr("transform", "translate(" + String(Number(click_pos[0]) - icon_size / 2) + "," + String(Number(click_pos[1]) - icon_size - 8) + ")")
                    .transition()
                    .duration(800)
                    .attr("transform", "translate(" + String(Number(click_pos[0]) - icon_size / 2) + "," + String(Number(click_pos[1]) - icon_size) + ")")
            }

            img_number = img_number + 1;
            // 让鼠标出现等待界面
            heatMap_svg_right.style("cursor", "wait");
            imageGrid_svg_right.style("cursor", "wait");
            imageTypeGrid_svg_right.style("cursor", "wait");
            var img_informations = await get_image_information_from_python(x = x, y = y, img_name = String(img_number),img_type = 1)
            // 只展示当前模型的信息
            console.log("点击热力图时候返回的信息： ", img_information)
            var model_id = "M1"
            const radioDNNButtons = document.querySelectorAll('input[name="radio_model"]');
            radioDNNButtons.forEach((radioButton) => {
                radioButton.addEventListener('click', function () {
                    if (this.checked) {
                        model_id = this.value;
                    }
                });
            });

            var img_information = {
                "label": img_informations['label'][model_id],
                "real_label": img_informations['real_label'][model_id],
                "img_robustness": img_informations['img_robustness'][model_id],
                "layer": img_informations['layer'][model_id],
                "layer_label":img_informations['layer_label'][model_id]
            }
            // 修改左边示例图
            if (click_number == 1) { //因为前面加一了
                d3.select("#click_start").attr("src", "../static/example/pic/" + String(img_number) + ".png?t=" + Math.random())
                d3.select("#cam_start").attr("src", "../static/example/pic/" + String(img_number) + "_cam.png?t=" + Math.random())

                var dataset_type = document.getElementById("select_dataset_type_selection").value
                if (dataset_type == "CIFAR10") {
                    // 显示类别信息
                    // document.getElementById("classified-as_1").innerText = cifar10_classes[parseInt(img_information['label'])]; //cifar10_classes数组在
                    document.getElementById("classified-as_1").innerText = `Pre : ${cifar10_classes[parseInt(img_information['label'])]}`;//cifar10_classes数组在image_type_grid.js中定义
                    document.getElementById("classified-as_1_real").innerText = `Real: ${cifar10_classes[parseInt(img_information['real_label'])]}`;//cifar10_classes数组在image_type_grid.js中定义
                    document.getElementById("classified-as_1").setAttribute('title', `Pre : ${cifar10_classes[parseInt(img_information['label'])]}`);//cifar10_classes数组在image_type_grid.js中定义
                    document.getElementById("classified-as_1_real").setAttribute('title', `Real: ${cifar10_classes[parseInt(img_information['real_label'])]}`);//cifar10_classes数组在image_type_grid.js中定义

                } else if (dataset_type == "SteeringAngle") {
                    // 显示类别信息
                    document.getElementById("classified-as_1").innerText = ((img_information['label'] - 0.5) * 160).toFixed(); //cifar10_classes数组在image_type_grid.js中定义
                } else if (dataset_type == "GTSRB") {
                    // 显示类别信息
                    // document.getElementById("classified-as_1").innerText = GTSRB_classes[parseInt(img_information['label']['M1'])];
                    document.getElementById("classified-as_1").innerText = `Pre : ${GTSRB_classes[parseInt(img_information['label'])]}`;//cifar10_classes数组在image_type_grid.js中定义
                    document.getElementById("classified-as_1_real").innerText = `Real: ${GTSRB_classes[parseInt(img_information['real_label'])]}`;//cifar10_classes数组在image_type_grid.js中定义
                    document.getElementById("classified-as_1").setAttribute('title', `Pre : ${GTSRB_classes[parseInt(img_information['label'])]}`);//cifar10_classes数组在image_type_grid.js中定义
                    document.getElementById("classified-as_1_real").setAttribute('title', `Real: ${GTSRB_classes[parseInt(img_information['real_label'])]}`);//cifar10_classes数组在image_type_grid.js中定义

                }
                // 显示鲁棒性
                document.getElementById("predicted-robustness-value_1").innerText = (img_information['img_robustness'] * 1000).toFixed(0);
                // 显示置信度
                show_confidence(img_information['layer'], img_information['layer_label'],dataset_type,"left")
            } else { // 右边相关
                d3.select("#click_end").attr("src", "../static/example/pic/" + String(img_number) + ".png?t=" + Math.random())
                d3.select("#cam_end").attr("src", "../static/example/pic/" + String(img_number) + "_cam.png?t=" + Math.random())

                var dataset_type = document.getElementById("select_dataset_type_selection").value
                if (dataset_type == "CIFAR10") {
                // 显示类别信息
                // document.getElementById("classified-as_1").innerText = cifar10_classes[parseInt(img_information['label']['M1'])]; //cifar10_classes数组在
                    document.getElementById("classified-as_2").innerText = `Pre : ${cifar10_classes[parseInt(img_information['label'])]}`;//cifar10_classes数组在image_type_grid.js中定义
                    document.getElementById("classified-as_2_real").innerText = `Real: ${cifar10_classes[parseInt(img_information['real_label'])]}`;//cifar10_classes数组在image_type_grid.js中定义
                    document.getElementById("classified-as_2").setAttribute('title', `Pre : ${cifar10_classes[parseInt(img_information['label'])]}`);//cifar10_classes数组在image_type_grid.js中定义
                    document.getElementById("classified-as_2_real").setAttribute('title', `Real: ${cifar10_classes[parseInt(img_information['real_label'])]}`);//cifar10_classes数组在image_type_grid.js中定义

                } else if (dataset_type == "SteeringAngle") {
                    // 显示类别信息
                    document.getElementById("classified-as_2").innerText = ((img_information['label']['M1'] - 0.5) * 160).toFixed(); //cifar10_classes数组在image_type_grid.js中定义
                }else if (dataset_type == "GTSRB") {
                    // 显示类别信息
                    // document.getElementById("classified-as_1").innerText = GTSRB_classes[parseInt(img_information['label']['M1'])];
                    document.getElementById("classified-as_2").innerText = `Pre : ${GTSRB_classes[parseInt(img_information['label'])]}`;//cifar10_classes数组在image_type_grid.js中定义
                    document.getElementById("classified-as_2_real").innerText = `Real: ${GTSRB_classes[parseInt(img_information['real_label'])]}`;//cifar10_classes数组在image_type_grid.js中定义
                    document.getElementById("classified-as_2").setAttribute('title', `Pre : ${GTSRB_classes[parseInt(img_information['label'])]}`);//cifar10_classes数组在image_type_grid.js中定义
                    document.getElementById("classified-as_2_real").setAttribute('title', `Real: ${GTSRB_classes[parseInt(img_information['real_label'])]}`);//cifar10_classes数组在image_type_grid.js中定义

                }
                // 显示鲁棒性
                document.getElementById("predicted-robustness-value_2").innerText = (img_information['img_robustness'] * 1000).toFixed(0);
                // 显示置信度
                show_confidence(img_information['layer'],img_information['layer_label'],dataset_type, "right")
            }
            //恢复成可以点击状态
            heatMap_svg_right.style("cursor", "pointer");
            imageGrid_svg_right.style("cursor", "pointer");
            imageTypeGrid_svg_right.style("cursor", "pointer");
        })

    } else { //取消点击操作
        heatMap_svg_right.style("cursor", "default");
        imageGrid_svg_right.style("cursor", "default");
        imageTypeGrid_svg_right.style("cursor", "default");
        // d3.select("#click_img_div")
        //     .style("display", "block")
        // d3.select("#compare_img_div")
        //     .style("display", "none")
        // d3.select("#div_model_name")
        //     .style("display", "flex")
        // d3.select("#div_compare_pic")
        //     .style("display", "none")

        // d3.selectAll(".div_imgcard_info").style("display", "flex")
        // d3.select("#seconde_type").style("display", "none")
        // d3.select("#seconde_robustness").style("display", "none")
        // heatMap_svg.select("#imageCompare_g").selectAll("line").remove()
        // heatMap_svg.select("#imageCompare_g").selectAll("g").remove()
        // heatMap_svg.select("#imageCompare_g").selectAll("circle").remove()
        heatMap_svg_right.on("click", null)
    }

})

// 橡皮擦事件
const clearBtn = document.querySelector("#btn_clear");
clearBtn.addEventListener("click", function () {
    //删除单机产生的结果~~~~~~~~~~
    //删除单击产生的图片
    heatMap_svg_right.select("#imageAndCircle_g_right")
        .selectAll("image")
        .remove()
    //删除点
    heatMap_svg_right.select("#imageAndCircle_g_right")
        .selectAll("circle")
        .remove()
    //删除划线对比产生的结果~~~~~~~~~~~
    //删除线
    heatMap_svg_right.select("#imageCompare_g_right")
        .selectAll("line")
        .remove()
    //删除点
    heatMap_svg_right.select("#imageCompare_g_right")
        .selectAll("circle")
        .remove()
    //删除图例
    heatMap_svg_right.select("#imageCompare_g_right")
        .selectAll("g")
        .remove()

    // 删除左边面板中的单点信息
    d3.select("#click_img").attr("src", "../static/example/initial_pic/None.png")
    d3.select("#cam_img").attr("src", "../static/example/initial_pic/None.png")

    // 显示类别信息
    document.getElementById("classified-as_1").innerText = ""; //cifar10_classes数组在image_type_grid.js中定义
    document.getElementById("classified-as_2").innerText = ""; //cifar10_classes数组在image_type_grid.js中定义
    // 显示鲁棒性
    document.getElementById("predicted-robustness-value_1").innerText = "";
    document.getElementById("predicted-robustness-value_2").innerText = "";


    // 删除左边面板中的图像对比信息
    d3.select("#click_start").attr("src", "../static/example/initial_pic/None.png")
    d3.select("#cam_start").attr("src", "../static/example/initial_pic/None.png")
    d3.select("#click_end").attr("src", "../static/example/initial_pic/None.png")
    d3.select("#cam_end").attr("src", "../static/example/initial_pic/None.png")

    // 显示类别信息
    document.getElementById("classified-as_1").innerText = ""; //cifar10_classes数组在image_type_grid.js中定义
    document.getElementById("classified-as_2").innerText = ""; //cifar10_classes数组在image_type_grid.js中定义
    // 显示鲁棒性
    document.getElementById("predicted-robustness-value_1").innerText = "";
    document.getElementById("predicted-robustness-value_2").innerText = "";

    if (lineBtn.checked == true) {
        // 显示置信度
        zero_confidece = [0, 0, 0, 0, 0, 0, 0, 0]
        zero_confidece_label = [0, 1, 2, 3, 4, 5, 6, 7]
        show_confidence(zero_confidece,zero_confidece_label, "left")
        show_confidence(zero_confidece,zero_confidece_label, "right")
    } else {
        // 显示置信度
        zero_confidece = [0, 0, 0, 0, 0, 0, 0, 0]
        zero_confidece_label = [0, 1, 2, 3, 4, 5, 6, 7]
        // show_confidence(zero_confidece, "left")
        // show_confidence(zero_confidece, "right")
        show_confidence_single(zero_confidece, zero_confidece_label,dataset_type)
    }


})

// 设置鲁棒性阈值事件~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
const setThresholdBtn = document.querySelector("#threshold_save")
setThresholdBtn.addEventListener("click", function () {
    change_heatMap();
})


// 中间工具栏-------------------------------------------------------------------------------------
// 切换显示的模型：包括鲁棒性分布和类别。
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
// 存储原本的颜色
const originalColor = radioDNNButtons[originalSelectedIndex].style.backgroundColor;

radioDNNButtons.forEach((radioButton) => {
    radioButton.addEventListener('click', function () {
        if (this.checked) {
            // 将上一个模型的划线对比清空（因为不同模型画同样的线意义不大）
            // 关闭lineBtn（先清空信息，然后再点击打开）
            if (lineBtn.checked == true) {
                lineBtn.click();
                lineBtn.click();
                // 在这里需要清除划线对比
                heatMap_svg.select("#imageCompare_g").selectAll("line").remove()
                heatMap_svg.select("#imageCompare_g").selectAll("g").remove()
                heatMap_svg.select("#imageCompare_g").selectAll("circle").remove()
            }
            var button_name_binding_dict = {
                "M1": ["select_model_selection"],
                "M2": ["select_model_selection_compare"],
                "M1-M2": ["select_model_selection", "select_model_selection_compare"]
            }
            // 当前单选按钮需要满足的模型选择器的内容，是否满足此次单选按钮的条件，如果不满足则回退到上一个单选按钮状态
            var selector = button_name_binding_dict[this.value];
            for (let i = 0; i < selector.length; i++) {
                var selected_model = document.getElementById(selector[i]).value;
                if (selected_model == "None") {
                    // 返回之前的选择
                    radioDNNButtons[originalSelectedIndex].checked = true;
                    // 获取 select 元素
                    const selectElement = document.getElementById(selector[i]);
                    // 将 select 元素的背景色改为红色
                    selectElement.style.backgroundColor = '#FFCCCC';
                    // 1 秒后将 select 元素的背景色改回原本的颜色
                    setTimeout(() => {
                        selectElement.style.backgroundColor = originalColor;
                    }, 1000);
                    return 0;
                }
            }
            // 把当前状态作为记录
            originalSelectedIndex = getCheckedIndex(radioDNNButtons);

            //绘制热力图
            change_heatMap()
            //绘制类别概览
            change_imageTypeGrid()
            change_matrix()


        }
    });
});



// 右边间工具栏-------------------------------------------------------------------------------------
//图片与图片类别切换按钮绑定
// const imageBtn_right = document.querySelector("#radio_image_right");
// const classificationBtn_right = document.querySelector("#radio_classification_right");
// const heatmapBtn_right = document.querySelector("#radio_heatmap_right");
// imageBtn_right.addEventListener("click", function () {
//     d3.select("#thresholds_right").style("display", "none");
//     d3.select("#types_right").style("display", "none");
//     d3.select("#svgContainer_imageGrid_right").style("display", "block");
//     d3.select("#svgContainer_typeGrid_right").style("display", "none");
//     d3.select("#svgContainer_heatMapGrid_right").style("display", "none");
//
// })
// classificationBtn_right.addEventListener("click", function () {
//     d3.select("#thresholds_right").style("display", "none");
//     d3.select("#types_right").style("display", "block");
//     d3.select("#svgContainer_imageGrid_right").style("display", "none");
//     d3.select("#svgContainer_typeGrid_right").style("display", "block");
//     d3.select("#svgContainer_heatMapGrid_right").style("display", "none");
//
// })
// heatmapBtn_right.addEventListener("click", function () {
//     d3.select("#thresholds_right").style("display", "block");
//     d3.select("#types_right").style("display", "none");
//     d3.select("#svgContainer_imageGrid_right").style("display", "none");
//     d3.select("#svgContainer_typeGrid_right").style("display", "none");
//     d3.select("#svgContainer_heatMapGrid_right").style("display", "block");
//
//
// })

// 中间工具栏-------------------------------------------------------------------------------------
//图片与图片类别切换按钮绑定
// const imageBtn = document.querySelector("#radio_image");
// const classificationBtn = document.querySelector("#radio_classification");
// const heatmapBtn = document.querySelector("#radio_heatmap");
// imageBtn.addEventListener("click", function () {
//     d3.select("#thresholds").style("display", "none");
//     d3.select("#types").style("display", "none");
//     d3.select("#svgContainer_imageGrid").style("display", "block");
//     d3.select("#svgContainer_typeGrid").style("display", "none");
//     d3.select("#svgContainer_heatMapGrid").style("display", "none");
//
// })
// classificationBtn.addEventListener("click", function () {
//     d3.select("#thresholds").style("display", "none");
//     d3.select("#types").style("display", "block");
//     d3.select("#svgContainer_imageGrid").style("display", "none");
//     d3.select("#svgContainer_typeGrid").style("display", "block");
//     d3.select("#svgContainer_heatMapGrid").style("display", "none");
//
// })
// heatmapBtn.addEventListener("click", function () {
//     d3.select("#thresholds").style("display", "block");
//     d3.select("#types").style("display", "none");
//     d3.select("#svgContainer_imageGrid").style("display", "none");
//     d3.select("#svgContainer_typeGrid").style("display", "none");
//     d3.select("#svgContainer_heatMapGrid").style("display", "block");
//
//
// })



const imageBtn = document.querySelector("#radio_image");
const classificationBtn = document.querySelector("#radio_classification");
imageBtn.addEventListener("click", function () {
    d3.select("#types").style("display", "flex");
    d3.select("#svgContainer_imageGrid_right").style("display", "block");
    d3.select("#svgContainer_typeGrid_right").style("display", "none");

})
classificationBtn.addEventListener("click", function () {
    d3.select("#types").style("display", "flex");
    d3.select("#svgContainer_imageGrid_right").style("display", "none");
    d3.select("#svgContainer_typeGrid_right").style("display", "block");
})

