/*
    这个js的功能主要是绘制类别网格
*/
//图片类别颜色
// 师兄原版颜色
// var labels_colors = { 0: '#8dd3c7', 1: '#ffffb3', 2: '#bebada', 3: '#fb8072', 4: '#80b1d3', 5: '#fdb462', 6: '#b3de69', 7: '#fccde5', 8: '#d9d9d9', 9: '#bc80bd', 10: "white"};
// colorbrewer颜色
var labels_colors = { 0: '#a6cee3', 1: '#1f78b4', 2: '#b2df8a', 3: '#33a02c', 4: '#fb9a99', 5: '#e31a1c', 6: '#fdbf6f', 7: '#ff7f00', 8: '#cab2d6', 9: '#6a3d9a', 10: "white" };
// var steeringAngle_labels_colors = { 0: '#a6cee3', 1: '#1f78b4', 2: '#b2df8a', 3: '#33a02c', 4: '#fb9a99', 5: '#e31a1c', 6: '#fdbf6f', 7: '#ff7f00', 8: '#cab2d6', 9: '#6a3d9a', 10: "white"};

//CIFAR10图片类别名字
const cifar10_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];
//SteeringAngle图片类别名字
const SteeringAngle_labels = ['-80~-64', '-64~-48', '-48~-32', '-32~-16', '-16~0', '0~16', '16~32', '32~48', '48~64', '64~80'];

//修改右边的网格类别概览
function change_imageTypeGrid() {
    // var imageTypeGrid_svg = d3.select("#imageTypeGrid_svg");
    // imageTyprGrid_svg.selectAll("rect").remove(); //先删除现有的
    draw_imageTypeGrid();
}


// 显示类别网格
function draw_imageTypeGrid() {
    var svg_typeGrid = d3.select("#imageTypeGrid_svg")
        .attr("width", imageGrid_width)
        .attr("height", imageGrid_height)
    var svg_typeGrid_right = d3.select("#imageTypeGrid_svg_right")
        .attr("width", imageGrid_width)
        .attr("height", imageGrid_height)

    var rectWidth = 40


    // 显示单选按钮选择的模型预测类别
    const selectedRadioModelButton = document.querySelector('input[name="radio_model"]:checked');
    const selectedValue = selectedRadioModelButton.value;
    var img_labels_lst_400;
    var img_fore_labels_lst_400;
    if (selectedValue != "M1-M2") {
        img_labels_lst_400 = img_DNN_output_lst_400_dict[selectedValue]
        img_fore_labels_lst_400 = img_DNN_for_output_lst_400_dict[selectedValue]
    } else {
        var type1 = img_DNN_output_lst_400_dict["M1"]
        var type2 = img_DNN_output_lst_400_dict["M2"]
        var type3 = img_DNN_for_output_lst_400_dict["M1"]
        var type4 = img_DNN_for_output_lst_400_dict["M2"]
        img_labels_lst_400 = type1.map((val, index) => parseInt(val) != parseInt(type2[index]) ? val : 10)
        img_fore_labels_lst_400 = type3.map((val, index) => parseInt(val) != parseInt(type4[index]) ? val : 10)

    }

    // 不同的数据集对应不同的数据操作
    var dataset_type = document.getElementById("select_dataset_type_selection").value //在Document.querySelector()通过id获取才需要加#
    console.log("dataset_type: ", dataset_type)
    if (dataset_type == "CIFAR10") {
        text_list = cifar10_classes
        labels_colors = { 0: '#a6cee3', 1: '#1f78b4', 2: '#b2df8a', 3: '#33a02c', 4: '#fb9a99', 5: '#e31a1c', 6: '#fdbf6f', 7: '#ff7f00', 8: '#cab2d6', 9: '#6a3d9a' };
        const colorBlocks = document.querySelectorAll('.color_block');
        for (let i = 0; i < colorBlocks.length; i++) {
            const colorIndex = i % Object.keys(labels_colors).length;
            const color = labels_colors[colorIndex];
            colorBlocks[i].style.backgroundColor = color;
        }
    } else if (dataset_type == "SteeringAngle") {
        text_list = SteeringAngle_labels
        labels_colors = { 0: '#543005', 1: '#8c510a', 2: '#bf812d', 3: '#dfc27d', 4: '#f6e8c3', 5: '#c7eae5', 6: '#80cdc1', 7: '#35978f', 8: '#01665e', 9: '#003c30' }
        const colorBlocks = document.querySelectorAll('.color_block');
        for (let i = 0; i < colorBlocks.length; i++) {
            const colorIndex = i % Object.keys(labels_colors).length;
            const color = labels_colors[colorIndex];
            colorBlocks[i].style.backgroundColor = color;
        }
        // console.log("img_labels_lst_400.length", img_labels_lst_400.length)
        // console.log("变换前img_labels_lst_400: ", img_labels_lst_400)
        // 从0~1变换到-80~80
        for (let index = 0; index < img_labels_lst_400.length; index++) {
            img_labels_lst_400[index] = (img_labels_lst_400[index] - 0.5) * 160
        }
        for (let index = 0; index < img_fore_labels_lst_400.length; index++) {
            img_fore_labels_lst_400[index] = (img_fore_labels_lst_400[index] - 0.5) * 160
        }
        // console.log("变换中img_labels_lst_400: ", img_labels_lst_400)
        // 对连续的值进行分箱处理
        for (let index = 0; index < img_labels_lst_400.length; index++) {
            for (let i = 0; i < text_list.length; i++) {
                const range = text_list[i].split('~').map(Number);
                if (img_labels_lst_400[index] >= range[0] && img_labels_lst_400[index] < range[1]) {
                    img_labels_lst_400[index] = i;
                    break;
                }
            }
        }
        for (let index = 0; index < img_fore_labels_lst_400.length; index++) {
            for (let i = 0; i < text_list.length; i++) {
                const range = text_list[i].split('~').map(Number);
                if (img_fore_labels_lst_400[index] >= range[0] && img_fore_labels_lst_400[index] < range[1]) {
                    img_fore_labels_lst_400[index] = i;
                    break;
                }
            }
        }
        // console.log("变换后img_labels_lst_400: ", img_labels_lst_400)
    }

    svg_typeGrid.select("#imageTypeGrid_g")
        .selectAll("rect")
        .data(img_labels_lst_400)
        .join("rect")
        .attr("number", (d, i) => i) //方便后续定位图片
        .attr("class", "class_rect")
        .attr("fill", function (d, i) {
            return labels_colors[parseInt(d)];
        })
        .attr("stroke", "black")
        .attr("height", rectWidth)
        .attr("width", rectWidth)
        .attr("x", (d, i) => rectWidth * (i % 20))
        .attr("y", (d, i) => rectWidth * Math.floor(i / 20))
    svg_typeGrid_right.select("#imageTypeGrid_g_right")
        .selectAll("rect")
        .data(img_fore_labels_lst_400)
        .join("rect")
        .attr("number", (d, i) => i) //方便后续定位图片
        .attr("class", "class_rect")
        .attr("fill", function (d, i) {
            return labels_colors[parseInt(d)];
        })
        .attr("stroke", "black")
        .attr("height", rectWidth)
        .attr("width", rectWidth)
        .attr("x", (d, i) => rectWidth * (i % 20))
        .attr("y", (d, i) => rectWidth * Math.floor(i / 20))

    //给标签上名字(右边部分)
    d3.select("#types_right")
        .selectAll(".classes_name")
        .each(function (d, i) {
            this.innerHTML = text_list[i];
        });
    d3.select("#types")
        .selectAll(".classes_name")
        .each(function (d, i) {
            this.innerHTML = text_list[i];
        });

}
