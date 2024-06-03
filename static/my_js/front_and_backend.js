/*
    这个js主要用来进行与后端python通信
*/

// ############################一些辅助函数################################
//通过比例尺获得特定位置的坐标，而不是网格的中心点了(这样能够保证平移的不变性)-------------------------------------
function calc_particular_coordinates(extent, bins, xScale, yScale) {
    // console.log("计算特定坐标时候的extent",extent)
    var particular_coordinates = [];
    var x_start = extent.start_x;
    var y_start = extent.start_y;
    var step_x = (extent.end_x - extent.start_x) / bins; //正数
    var step_y = (extent.end_y - extent.start_y) / bins; //负数
    // 让起点始终为能够整除步长的数
    // console.log("step_x: ", step_x)
    // console.log("step_y: ", step_y)
    // console.log("x_start % step_x: ", x_start % step_x)
    // console.log("前x_start: ", x_start)
    // console.log("前y_start: ", y_start)
    if (x_start % step_x != 0) {
        var devide_number = x_start / step_x;
        var integer_number = parseInt(devide_number);
        // console.log("x integer_number: ", integer_number);
        var float_number = devide_number - integer_number;
        // console.log("x float_number: ", float_number);
        // 一个数学规律(让坐标发生一点点的小偏移，使得满足平移不变性)
        if (devide_number <= 0) {
            x_start -= float_number * step_x;
        } else {
            x_start += step_x;
            x_start -= float_number * step_x
        }
    }
    if (y_start % step_y != 0) {
        var devide_number = y_start / step_y;
        var integer_number = parseInt(devide_number);
        // console.log("y integer_number: ", integer_number);
        var float_number = devide_number - integer_number;
        // console.log("y float_number: ", float_number);
        // 一个数学规律(让坐标发生一点点的小偏移，使得满足平移不变性)
        if (devide_number <= 0) {
            y_start -= float_number * step_y;
        } else {
            y_start += step_y;
            y_start -= float_number * step_y
        }
    }
    // console.log("后x_start: ", x_start)
    // console.log("后y_start: ", y_start)
    for (var i = 0; i < bins; i++) {
        for (var j = 0; j < bins; j++) { //注意是先遍历x
            particular_x = x_start + j * step_x;
            particular_y = y_start + i * step_y;
            particular_coordinates.push([particular_x, particular_y]);
        }
    }
    coordinates_offset["x"] = x_start - extent.start_x;
    coordinates_offset["y"] = y_start - extent.start_y;

    return particular_coordinates;
}
// ############################一些辅助函数################################


//从后端获取对应分辨率坐标的鲁棒性值、图片类别概览中的类别，以及每一张图片对应的坐标
async function get_information_from_python(extent, xScale, yScale) {
    //让等待图表转起来
    var spinner = d3.selectAll(".spinner-border");
    spinner.style("display", "block");
    // 计算坐标，起点为左上角
    var coordinates = calc_particular_coordinates(extent, bins, xScale, yScale);
    var idw_p = document.getElementById("idw_p").value; //获取反距离系数
    // console.log("coordinates: ", coordinates)
    await axios.post("/get_information_data", {"coordinates": coordinates, "idw_p":idw_p})
        .then(function (response) {
            data = response.data;
            // robustness_dict = data.robustness_dict;
            // console.log("robustness_dict: ", robustness_dict)
            confidence_dict = data.confidence_dict;
            confidence_fore_dict = data.confidence_fore_dict;
            img_DNN_for_output_lst_400_dict = data.img_DNN_for_output_lst_400_dict;
            img_DNN_output_lst_400_dict = data.img_DNN_output_lst_400_dict;
            img_coords_lst_400_dict = data.img_coords_lst_400_dict;
            conf_matrix_dic = data.conf_matrix_dic
            // console.log("get_information_from_python中img_types_lst_400_dict：",img_DNN_output_lst_400_dict)
        })
    return 0;
}
//用户在点击鲁棒性地图上的某个坐标后，需要后端返回相应的信息
async function get_image_information_from_python(x, y, img_name = "one",img_type) {
    //让等待图表转起来
    var spinner = d3.selectAll(".spinner-border");
    spinner.style("display", "block");
    var points = [x, y]
    var label, img_robustness, layer
    await axios.post("/get_image_information", { "points": points, "img_name": img_name,"img_type": img_type})
        .then(function (response) {
            data = response.data;
            //让等待图表隐藏起来
            spinner.style("display", "none");
        })
    return data;
}

async function evaluate_image_from_python(img_number, img_name, img_type){
    //让等待图表转起来
    var spinner = d3.selectAll(".spinner-border");
    spinner.style("display", "block");
    var label, img_robustness, layer
    await axios.post("/evaluate_image", { "img_number": img_number, "img_name": img_name, "img_type":img_type})
        .then(function (response) {
            data = response.data;
            //让等待图表隐藏起来
            spinner.style("display", "none");
        })
    return data;
}

//通知后端准备所有模型的共享的数据
async function prepare_shared_data(dataset_type){
    //让等待图表转起来
    var spinner = d3.selectAll(".spinner-border");
    spinner.style("display", "block");
    await axios.post("/prepare_shared_data", {"dataset_type":dataset_type});
    return 0;
}

//通知后端准备单个模型的数据
async function prepare_DNN_data(model_id, model_name){
    //让等待图表转起来
    var spinner = d3.selectAll(".spinner-border");
    spinner.style("display", "block");
    await axios.post("/prepare_DNN_data", {"model_id": model_id, "model_name": model_name});
    //让等待图表转停止
    var spinner = d3.selectAll(".spinner-border");
    spinner.style("display", "none");
    return 0;
}











