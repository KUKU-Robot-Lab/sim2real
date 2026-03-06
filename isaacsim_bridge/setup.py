from setuptools import setup


package_name = "isaacsim_bridge"


setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/isaacsim_bridge.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@example.com",
    description="Bridge Isaac Sim topics to OpenArm and Tesollo real controllers.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "bridge_node = isaacsim_bridge.bridge_node:main",
            "joint_error_recorder = isaacsim_bridge.joint_error_recorder:main",
            "joint_tuning_report = isaacsim_bridge.joint_tuning_report:main",
            "joint_tuning_cycle = isaacsim_bridge.joint_tuning_cycle:main",
        ],
    },
)
